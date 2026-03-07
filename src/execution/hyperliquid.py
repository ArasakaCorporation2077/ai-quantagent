"""Hyperliquid perpetual futures execution.

Executes trading signals on Hyperliquid exchange.
Supports dry-run mode (default) for safety.
"""

import logging
import time

from eth_account import Account
from hyperliquid.exchange import Exchange
from hyperliquid.info import Info
from hyperliquid.utils import constants
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# Map our symbol names to Hyperliquid coin names
SYMBOL_TO_COIN = {
    'BTCUSDT': 'BTC',
    'ETHUSDT': 'ETH',
    'SOLUSDT': 'SOL',
    'XRPUSDT': 'XRP',
    'DOGEUSDT': 'DOGE',
    'BNBUSDT': 'BNB',
    'ADAUSDT': 'ADA',
    'LINKUSDT': 'LINK',
    'AVAXUSDT': 'AVAX',
    'DOTUSDT': 'DOT',
    'LTCUSDT': 'LTC',
    'SUIUSDT': 'SUI',
    'NEARUSDT': 'NEAR',
    'APTUSDT': 'APT',
    'UNIUSDT': 'UNI',
    'TONUSDT': 'TON',
    'XLMUSDT': 'XLM',
    'EOSUSDT': 'EOS',
    'BCHUSDT': 'BCH',
    '1000SHIBUSDT': 'kSHIB',
}

MIN_ORDER_USD = 11.0  # Hyperliquid minimum ~$10.25, use $11 for safety

# Per-coin leverage settings (cross margin)
COIN_LEVERAGE = {
    'BTC': 3,
    'ETH': 3,
    'SOL': 2,
    'XRP': 2,
    'DOGE': 2,
}
DEFAULT_LEVERAGE = 2


class HyperliquidExecutor:
    def __init__(self, secrets: dict):
        hl_cfg = secrets.get('hyperliquid', {})
        self.account_address = hl_cfg.get('account_address', '')
        secret_key = hl_cfg.get('secret_key', '')

        if not self.account_address or not secret_key:
            raise ValueError('Hyperliquid credentials not found. Set HYPERLIQUID_ACCOUNT_ADDRESS and HYPERLIQUID_SECRET_KEY in .env')

        self.wallet = Account.from_key(secret_key)
        self.exchange = Exchange(self.wallet, constants.MAINNET_API_URL)
        self.info = Info(constants.MAINNET_API_URL, skip_ws=True)

    def get_positions(self) -> dict[str, float]:
        """Get current positions as {coin: size_usd}. Positive=long, negative=short.
        Uses current mid price (not entry price) for accurate exposure calculation.
        """
        state = self.info.user_state(self.account_address)
        all_mids = self.info.all_mids()
        positions = {}

        for asset_pos in state.get('assetPositions', []):
            pos = asset_pos.get('position', {})
            coin = pos.get('coin', '')
            szi = float(pos.get('szi', 0))

            if szi != 0:
                mid_price = float(all_mids.get(coin, 0))
                if mid_price > 0:
                    positions[coin] = szi * mid_price

        return positions

    def get_account_value(self) -> float:
        """Get total account value in USD."""
        state = self.info.user_state(self.account_address)
        margin = state.get('marginSummary', {})
        return float(margin.get('accountValue', 0))

    def execute_signal(self, target_positions: dict[str, float],
                       dry_run: bool = True) -> dict:
        """Execute rebalancing from current positions to target positions.

        Args:
            target_positions: {symbol: usd_amount} from signal generator
                              Positive=long, negative=short
            dry_run: If True, only print plan without executing

        Returns:
            dict with execution results
        """
        # 1. Get current positions
        current = self.get_positions()

        # 2. Convert target symbols to coins
        target_coins = {}
        for sym, usd_amt in target_positions.items():
            coin = SYMBOL_TO_COIN.get(sym, sym.replace('USDT', ''))
            target_coins[coin] = usd_amt

        # 3. Compute deltas
        all_coins = set(list(current.keys()) + list(target_coins.keys()))
        orders = []

        for coin in sorted(all_coins):
            cur = current.get(coin, 0)
            tgt = target_coins.get(coin, 0)
            delta = tgt - cur

            if abs(delta) < MIN_ORDER_USD:
                continue

            orders.append({
                'coin': coin,
                'current_usd': cur,
                'target_usd': tgt,
                'delta_usd': delta,
                'side': 'BUY' if delta > 0 else 'SELL',
            })

        # 4. Print plan
        self._print_plan(current, target_coins, orders, dry_run)

        # 5. Execute if not dry run
        results = []
        if not dry_run and orders:
            # Set leverage for each coin before ordering
            self._set_leverage([o['coin'] for o in orders])
            console.print('\n[bold yellow]Executing orders...[/bold yellow]\n')
            for order in orders:
                result = self._execute_order_with_retry(order)
                results.append(result)

        return {
            'dry_run': dry_run,
            'orders_planned': len(orders),
            'orders_executed': len(results),
            'results': results,
        }

    def _execute_order_with_retry(self, order: dict, max_retries: int = 5) -> dict:
        """Execute order with retry on rate limit errors."""
        for attempt in range(max_retries):
            result = self._execute_order(order)
            if result['status'] != 'rate_limited':
                return result
            wait = 15 * (attempt + 1)
            console.print(f'  [dim]Rate limited, waiting {wait}s... ({attempt+1}/{max_retries})[/dim]')
            time.sleep(wait)
        return result

    def _execute_order(self, order: dict) -> dict:
        """Execute a single market order."""
        coin = order['coin']
        delta_usd = abs(order['delta_usd'])
        is_buy = order['delta_usd'] > 0

        try:
            # Get current price to calculate size in coins
            price = self._get_mid_price(coin)
            if price <= 0:
                return {'coin': coin, 'status': 'error', 'msg': 'Could not get price'}

            sz = delta_usd / price

            # Round size to appropriate precision
            sz = self._round_size(coin, sz)
            if sz <= 0:
                return {'coin': coin, 'status': 'skipped', 'msg': 'Size too small after rounding'}

            result = self.exchange.market_open(
                name=coin,
                is_buy=is_buy,
                sz=sz,
            )

            status = result.get('status', 'unknown') if isinstance(result, dict) else 'sent'
            side = 'BUY' if is_buy else 'SELL'

            if status == 'err':
                err_msg = str(result.get('response', result))
                if 'Too many cumulative requests' in err_msg:
                    return {'coin': coin, 'status': 'rate_limited', 'side': side, 'msg': err_msg}
                console.print(f'  [red]{side} {coin} sz={sz:.6f} (~${delta_usd:.0f}) -> ERROR: {err_msg}[/red]')
                return {'coin': coin, 'status': 'error', 'side': side, 'msg': err_msg}

            console.print(f'  {side} {coin} sz={sz:.6f} (~${delta_usd:.0f}) -> {status}')
            return {'coin': coin, 'status': status, 'side': side, 'sz': sz, 'usd': delta_usd}

        except Exception as e:
            console.print(f'  [red]ERROR {coin}: {e}[/red]')
            return {'coin': coin, 'status': 'error', 'msg': str(e)}

    def _set_leverage(self, coins: list[str]):
        """Set cross-margin leverage for each coin before trading."""
        seen = set()
        for coin in coins:
            if coin in seen:
                continue
            seen.add(coin)
            lev = COIN_LEVERAGE.get(coin, DEFAULT_LEVERAGE)
            try:
                self.exchange.update_leverage(lev, coin, is_cross=True)
                console.print(f'  Leverage {coin}: {lev}x cross')
            except Exception as e:
                console.print(f'  [red]Leverage {coin} failed: {e}[/red]')

    def _get_mid_price(self, coin: str) -> float:
        """Get current mid price for a coin."""
        try:
            all_mids = self.info.all_mids()
            return float(all_mids.get(coin, 0))
        except Exception:
            return 0.0

    def _round_size(self, coin: str, sz: float) -> float:
        """Round order size to exchange-acceptable precision."""
        meta = self.info.meta()
        for asset in meta.get('universe', []):
            if asset.get('name') == coin:
                sz_decimals = asset.get('szDecimals', 6)
                return round(sz, sz_decimals)
        # Default: 6 decimals
        return round(sz, 6)

    def _print_plan(self, current: dict, target: dict, orders: list, dry_run: bool):
        mode = '[bold red]LIVE EXECUTION[/bold red]' if not dry_run else '[bold yellow]DRY RUN[/bold yellow]'
        console.print(f'\n=== Hyperliquid Execution Plan ({mode}) ===\n')

        # Current positions
        if current:
            console.print('[dim]Current Positions:[/dim]')
            for coin, usd in sorted(current.items(), key=lambda x: x[1]):
                direction = 'LONG' if usd > 0 else 'SHORT'
                console.print(f'  {coin}: {direction} ${abs(usd):.0f}')
        else:
            console.print('[dim]Current Positions: (none)[/dim]')

        console.print()

        # Orders table
        if orders:
            t = Table(title='Orders to Execute')
            t.add_column('#', style='dim')
            t.add_column('Side', justify='center')
            t.add_column('Coin', style='cyan')
            t.add_column('Lev', justify='center', style='yellow')
            t.add_column('Current', justify='right')
            t.add_column('Target', justify='right')
            t.add_column('Delta', justify='right')

            for i, o in enumerate(orders, 1):
                side_style = '[green]BUY[/green]' if o['side'] == 'BUY' else '[red]SELL[/red]'
                lev = COIN_LEVERAGE.get(o['coin'], DEFAULT_LEVERAGE)
                t.add_row(
                    str(i),
                    side_style,
                    o['coin'],
                    f'{lev}x',
                    f'${o["current_usd"]:+,.0f}',
                    f'${o["target_usd"]:+,.0f}',
                    f'${o["delta_usd"]:+,.0f}',
                )
            console.print(t)
        else:
            console.print('[green]No orders needed - positions already aligned.[/green]')

        if dry_run:
            console.print('\n[yellow]This is a DRY RUN. Add --confirm to execute.[/yellow]\n')

    def print_positions(self):
        """Print current positions nicely."""
        current = self.get_positions()
        account_value = self.get_account_value()

        console.print(f'\n[bold]Hyperliquid Account[/bold]')
        console.print(f'Address: {self.account_address[:10]}...{self.account_address[-6:]}')
        console.print(f'Account Value: ${account_value:,.2f}\n')

        if not current:
            console.print('[dim]No open positions[/dim]\n')
            return

        t = Table(title='Open Positions')
        t.add_column('Coin', style='cyan')
        t.add_column('Direction', justify='center')
        t.add_column('Size (USD)', justify='right')

        total_long = 0
        total_short = 0

        for coin, usd in sorted(current.items(), key=lambda x: abs(x[1]), reverse=True):
            direction = '[green]LONG[/green]' if usd > 0 else '[red]SHORT[/red]'
            t.add_row(coin, direction, f'${abs(usd):,.0f}')
            if usd > 0:
                total_long += usd
            else:
                total_short += usd

        console.print(t)
        console.print(f'\n  Long:  ${total_long:+,.0f}')
        console.print(f'  Short: ${total_short:+,.0f}')
        console.print(f'  Net:   ${total_long + total_short:+,.0f}\n')
