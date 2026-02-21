# AI Quantagent

AI Agent Hedge Fund - Automated crypto futures trading with pre-built quant alphas

[English](#overview) | [한국어](#개요)

---

## Overview

A dollar-neutral crypto futures trading system that runs on Hyperliquid. Includes 9 pre-built alpha signals (backtested on 2022-2026 data, Sharpe >= 0.5). Just add your Hyperliquid wallet and start trading.

### How It Works

```
1. Signal Generation : Pre-built alphas evaluate latest market data → daily trading signals
2. Portfolio Build   : Combines 9 alphas via Sharpe-proportional weighting
3. Auto Execution    : Automatic rebalancing on Hyperliquid (cron)
```

### Strategy

- **Dollar Neutral**: Long exposure = Short exposure (market direction independent)
- **Cross-sectional**: Long/short based on relative strength between coins
- **Multi-alpha**: 9 alphas combined via Sharpe-proportional weighting
- **Daily Rebalance**: Automatic rebalancing at UTC 00:00 (KST 09:00)

### Backtest Results (2022-2026)

- **Annual Return**: ~25%
- **Sharpe Ratio**: 1.01
- **Max Drawdown**: -21%
- **Strategy**: Dollar-neutral, 10 coins, daily rebalance

## Quick Start

### 1. Install

```bash
git clone https://github.com/ArasakaCorporation2077/ai-quantagent.git
cd ai-quantagent
pip install -r requirements.txt
```

### 2. Configure

```bash
cp .env.example .env
# Edit .env with your Hyperliquid keys
```

**Required keys:**
- `HYPERLIQUID_ACCOUNT_ADDRESS`: Your Hyperliquid wallet address
- `HYPERLIQUID_SECRET_KEY`: Your Agent Wallet private key

### 3. Set Capital

Edit `config/config.yaml`:

```yaml
execution:
  capital: 800   # Change this to your amount (in USD)
```

| Your Capital | Recommended `capital` | Leverage | Note |
|-------------|----------------------|----------|------|
| $200~$500 | Same as your balance | 2x | Minimum viable, ~80% margin usage |
| $500~$2,000 | 1.5~2x your balance | 2~3x | Good starting range |
| $2,000~$10,000 | 1.5~2x your balance | 2~3x | Comfortable |
| $10,000+ | 1.5x your balance | 2~3x | Consider adding more symbols |

**Example**: $1,000 balance → set `capital: 1500~2000` with 2x leverage (~75~100% margin usage)

Leverage settings in `src/execution/hyperliquid.py`:
```python
COIN_LEVERAGE = {
    'BTC': 3,   # BTC, ETH: 3x (lower volatility)
    'ETH': 3,
}
DEFAULT_LEVERAGE = 2  # Others: 2x
```

### 4. Run

```bash
python main.py signal              # View today's signal
python main.py execute             # Dry run (preview orders)
python main.py execute --confirm   # Execute on Hyperliquid
python main.py positions           # Check current positions
```

### 5. Automate (cron)

```bash
# Daily rebalance at UTC 00:00 (KST 09:00)
crontab -e
0 0 * * * cd /path/to/ai-quantagent && python3 main.py execute --confirm >> logs/rebalance.log 2>&1
```

## Commands

| Command | Description |
|---------|-------------|
| `signal` | Generate live trading signal from combined alphas |
| `execute` | Generate signal and execute on Hyperliquid |
| `positions` | Show current Hyperliquid positions |
| `report` | Show included alphas and their performance |
| `combine` | Combine alphas into a portfolio and compare methods |
| `backtest` | Backtest a single alpha expression |
| `download` | Download historical kline data from Binance |
| `process` | Process raw CSVs into enriched parquet files |

## How It Makes Money

This bot profits from **relative performance differences** between coins, not market direction:

```
Market goes UP:
  Long coins rise +8%, Short coins rise +3% → Profit: +5%

Market goes DOWN:
  Long coins fall -3%, Short coins fall -8% → Profit: +5%

Loss scenario:
  Long coins underperform Short coins → Loss
```

Since Long = Short (dollar neutral), the portfolio is hedged against market-wide moves.

## Architecture

```
src/
├── alpha/          # Alpha parser, evaluator, validator, transforms (42 functions)
├── backtest/       # Backtesting engine, metrics (Sharpe/Sortino/etc), position sizing
├── data/           # Binance data downloader & processor
├── execution/      # Hyperliquid order execution with retry & leverage management
├── orchestrator/   # Alpha combiner, signal generator
├── storage/        # SQLite database for strategies & results
└── config.py       # Configuration loader
```

## Disclaimer

This is an experimental research project. Use at your own risk. Past backtest performance does not guarantee future results. Start with small capital.

---

# 한국어

## 개요

Hyperliquid에서 운용하는 달러 뉴트럴 크립토 선물 트레이딩 시스템입니다. 검증된 9개 알파 시그널이 포함되어 있어 (2022~2026 백테스트, Sharpe >= 0.5), Hyperliquid 지갑만 연결하면 바로 트레이딩할 수 있습니다.

### 작동 방식

```
1. 시그널 생성   : 내장된 알파가 최신 시장 데이터 분석 → 일일 매매 시그널
2. 포트폴리오    : 9개 알파를 Sharpe 비례 가중평균으로 조합
3. 자동 실행     : Hyperliquid에서 자동 리밸런싱 (cron)
```

### 전략

- **달러 뉴트럴**: 롱 비중 = 숏 비중 (시장 방향과 무관하게 수익 추구)
- **횡단면 분석**: 코인 간 상대 강도로 롱/숏 결정
- **멀티 알파**: 9개 알파를 Sharpe 비례 가중평균으로 조합
- **일일 리밸런싱**: 매일 오전 9시(한국시간) 자동 리밸런싱

### 백테스트 성과 (2022~2026)

- **연간 수익률**: ~25%
- **Sharpe 비율**: 1.01
- **최대 낙폭**: -21%
- **전략**: 달러 뉴트럴, 10개 코인, 일일 리밸런싱

## 빠른 시작

### 1. 설치

```bash
git clone https://github.com/ArasakaCorporation2077/ai-quantagent.git
cd ai-quantagent
pip install -r requirements.txt
```

### 2. 설정

```bash
cp .env.example .env
# .env 파일에 Hyperliquid 키 입력
```

**필수 키:**
- `HYPERLIQUID_ACCOUNT_ADDRESS`: Hyperliquid 지갑 주소
- `HYPERLIQUID_SECRET_KEY`: Agent Wallet 비밀키

### 3. 자본금 설정

`config/config.yaml` 수정:

```yaml
execution:
  capital: 800   # 본인 금액으로 변경 (USD)
```

| 계좌 잔고 | 권장 `capital` 설정 | 레버리지 | 비고 |
|----------|-------------------|---------|------|
| $200~$500 | 잔고와 동일 | 2배 | 최소 운용 금액, 마진 사용률 ~80% |
| $500~$2,000 | 잔고의 1.5~2배 | 2~3배 | 권장 시작 금액 |
| $2,000~$10,000 | 잔고의 1.5~2배 | 2~3배 | 안정적 운용 |
| $10,000 이상 | 잔고의 1.5배 | 2~3배 | 심볼 추가 고려 |

**예시**: $1,000 잔고 → `capital: 1500~2000` 설정 (2배 레버리지 기준 마진 사용률 75~100%)

레버리지 설정 변경: `src/execution/hyperliquid.py`
```python
COIN_LEVERAGE = {
    'BTC': 3,   # BTC, ETH: 3배 (변동성 낮음)
    'ETH': 3,
}
DEFAULT_LEVERAGE = 2  # 나머지: 2배
```

### 4. 실행

```bash
python main.py signal              # 오늘의 시그널 확인
python main.py execute             # 드라이런 (주문 미리보기)
python main.py execute --confirm   # 실제 주문 실행
python main.py positions           # 현재 포지션 확인
```

### 5. 자동화 (cron)

```bash
# 매일 오전 9시 (한국시간) 자동 리밸런싱
crontab -e
0 0 * * * cd /path/to/ai-quantagent && python3 main.py execute --confirm >> logs/rebalance.log 2>&1
```

## 수익 구조

이 봇은 **시장 방향과 무관하게** 코인 간 **상대적 성과 차이**로 수익을 냅니다:

```
시장 상승 시:
  롱 코인 +8% 상승, 숏 코인 +3% 상승 → 수익: +5%

시장 하락 시:
  롱 코인 -3% 하락, 숏 코인 -8% 하락 → 수익: +5%

손실 조건:
  롱 코인이 숏 코인보다 못할 때
```

롱 = 숏 (달러 뉴트럴)이므로 시장 전체 등락에 대한 헤지가 되어 있습니다.

## 명령어 목록

| 명령어 | 설명 |
|--------|------|
| `signal` | 조합된 알파에서 라이브 시그널 생성 |
| `execute` | 시그널 생성 후 Hyperliquid에서 실행 |
| `positions` | 현재 Hyperliquid 포지션 확인 |
| `report` | 포함된 알파와 성과 확인 |
| `combine` | 알파를 포트폴리오로 조합하고 방법 비교 |
| `backtest` | 단일 알파 수식 백테스트 |
| `download` | Binance 과거 캔들 데이터 다운로드 |
| `process` | CSV를 parquet으로 가공 |

## 주의사항

이 프로젝트는 실험적 연구 목적입니다. 투자 손실에 대한 책임은 본인에게 있습니다. 과거 백테스트 성과가 미래 수익을 보장하지 않습니다. 소액으로 시작하세요.

## License

MIT
