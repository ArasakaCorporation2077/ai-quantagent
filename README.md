# AI Quantagent

AI Agent Hedge Fund - LLM 기반 자동 알파 발굴 및 크립토 선물 트레이딩 시스템

## Overview

GPT/Claude를 활용해 퀀트 알파를 자동 발굴하고, 백테스트로 검증한 뒤, Hyperliquid에서 달러 뉴트럴 포트폴리오를 자동 운용하는 시스템입니다.

### How It Works

```
1. Alpha Discovery   : LLM이 트레이딩 전략 아이디어 생성 → 수학 수식으로 변환
2. Backtesting       : 과거 데이터로 수익성 검증 (Sharpe >= 0.5 필터)
3. Signal Generation : 검증된 알파 조합 → 일일 매매 시그널 생성
4. Auto Execution    : Hyperliquid에서 자동 리밸런싱 (cron)
```

### Strategy

- **Dollar Neutral**: Long exposure = Short exposure (시장 방향 무관)
- **Cross-sectional**: 코인 간 상대 강도로 롱/숏 결정
- **Multi-alpha**: 여러 알파를 Sharpe 비례 가중평균으로 조합
- **Daily Rebalance**: 매일 UTC 00:00 (KST 09:00) 자동 리밸런싱

## Quick Start

### 1. Install

```bash
git clone https://github.com/YOUR_USERNAME/ai-quantagent.git
cd ai-quantagent
pip install -r requirements.txt
```

### 2. Configure

```bash
cp config/secrets.yaml.example config/secrets.yaml
# Edit secrets.yaml with your API keys
```

**Required keys:**
- `openai_api_key`: OpenAI API key (for alpha discovery)
- `hyperliquid.account_address`: Hyperliquid wallet address
- `hyperliquid.secret_key`: Hyperliquid Agent Wallet private key

### 3. Download & Process Data

```bash
python main.py download    # Download Binance Futures klines
python main.py process     # Process CSVs into parquet
```

### 4. Discover Alphas

```bash
python main.py pipeline --iterations 10
```

This uses GPT to generate alpha ideas, converts them to math expressions, and backtests each one. Takes ~1 hour for 10 iterations.

### 5. Generate Signal & Execute

```bash
python main.py signal              # View today's signal
python main.py execute             # Dry run (preview orders)
python main.py execute --confirm   # Execute on Hyperliquid
python main.py positions           # Check current positions
```

### 6. Automate (cron)

```bash
# Daily rebalance at UTC 00:00 (KST 09:00)
crontab -e
0 0 * * * cd /path/to/ai-quantagent && python3 main.py execute --confirm >> logs/rebalance.log 2>&1
```

## Commands

| Command | Description |
|---------|-------------|
| `download` | Download historical kline data from Binance |
| `process` | Process raw CSVs into enriched parquet files |
| `pipeline` | Run the full alpha discovery pipeline (LLM) |
| `backtest` | Backtest a single alpha expression |
| `combine` | Combine top alphas into a portfolio |
| `signal` | Generate live trading signal |
| `execute` | Generate signal and execute on Hyperliquid |
| `positions` | Show current Hyperliquid positions |
| `report` | Show top discovered alphas from DB |

## Architecture

```
src/
├── alpha/          # Alpha parser, evaluator, validator, transforms (42 functions)
├── backtest/       # Backtesting engine, metrics (Sharpe/Sortino/etc), position sizing
├── data/           # Binance data downloader & processor
├── execution/      # Hyperliquid order execution
├── llm/            # LLM client (OpenAI/Anthropic), prompts, response parser
├── orchestrator/   # Pipeline loop, alpha combiner, signal generator
├── storage/        # SQLite database for strategies & results
└── config.py       # Configuration loader
```

## Configuration

Edit `config/config.yaml`:

- **symbols**: Coins to trade (default: 10 major coins)
- **execution.capital**: Total capital in USD
- **execution.symbols**: Coins for live trading
- **backtest.min_sharpe**: Minimum Sharpe ratio to keep an alpha
- **llm.provider**: `openai` or `anthropic`

## Leverage Settings

Default leverage in `src/execution/hyperliquid.py`:

| Coin | Leverage |
|------|----------|
| BTC | 3x cross |
| ETH | 3x cross |
| Others | 2x cross |

## Key Design Decisions

- **AST-based evaluation**: No `eval()` - safe recursive descent parser for alpha expressions
- **MultiIndex panel**: (symbol, timestamp) for clean time-series/cross-section separation
- **Free data only**: Binance Vision public klines, no paid data needed
- **42 transform functions**: ts_mean, ts_zscore, cs_rank, decay_linear, etc.
- **Deduplication**: SHA256 hash on expression string to avoid duplicates

## Disclaimer

This is an experimental research project. Use at your own risk. Past backtest performance does not guarantee future results. Start with small capital.

## License

MIT
