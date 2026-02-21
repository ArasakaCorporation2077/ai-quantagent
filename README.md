# AI Quantagent

AI Agent Hedge Fund - LLM-powered automated alpha discovery & crypto futures trading system

[English](#overview) | [한국어](#개요)

---

## Overview

A system that uses GPT/Claude to automatically discover quant alphas, validates them through backtesting, and runs a dollar-neutral portfolio on Hyperliquid.

### How It Works

```
1. Alpha Discovery   : LLM generates trading strategy ideas → converts to math expressions
2. Backtesting       : Validates profitability on historical data (Sharpe >= 0.5 filter)
3. Signal Generation : Combines proven alphas → generates daily trading signals
4. Auto Execution    : Automatic rebalancing on Hyperliquid (cron)
```

### Strategy

- **Dollar Neutral**: Long exposure = Short exposure (market direction independent)
- **Cross-sectional**: Long/short based on relative strength between coins
- **Multi-alpha**: Multiple alphas combined via Sharpe-proportional weighting
- **Daily Rebalance**: Automatic rebalancing at UTC 00:00 (KST 09:00)

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
# Edit .env with your API keys
```

**Required keys:**
- `OPENAI_API_KEY`: OpenAI API key (only needed for alpha discovery)
- `HYPERLIQUID_ACCOUNT_ADDRESS`: Hyperliquid wallet address
- `HYPERLIQUID_SECRET_KEY`: Hyperliquid Agent Wallet private key

### 3. Quick Start (Use Pre-built Alphas)

The repo includes a pre-built database with 9 proven alphas (Sharpe >= 0.5, backtested on 2022-2026 data). You can skip data download and alpha discovery, and go straight to trading:

```bash
python main.py report              # View included alphas
python main.py signal              # Generate today's signal
python main.py execute             # Dry run
python main.py execute --confirm   # Live execute on Hyperliquid
```

### 4. (Optional) Discover Your Own Alphas

If you want to find additional alphas, you need an OpenAI API key:

```bash
python main.py download                  # Download Binance Futures klines
python main.py process                   # Process CSVs into parquet
python main.py pipeline --iterations 10  # ~1 hour, uses GPT
```

### 5. Set Capital

Edit `config/config.yaml` to set your capital:

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

**Example**: If you have $1,000 in your Hyperliquid account, set `capital: 1500~2000` with 2x leverage. This uses ~75~100% of your margin.

Leverage settings are in `src/execution/hyperliquid.py`:
```python
COIN_LEVERAGE = {
    'BTC': 3,   # BTC, ETH: 3x (lower volatility)
    'ETH': 3,
}
DEFAULT_LEVERAGE = 2  # Others: 2x
```

### 6. Generate Signal & Execute

```bash
python main.py signal              # View today's signal
python main.py execute             # Dry run (preview orders)
python main.py execute --confirm   # Execute on Hyperliquid
python main.py positions           # Check current positions
```

### 7. Automate (cron)

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

## Key Design Decisions

- **AST-based evaluation**: No `eval()` - safe recursive descent parser for alpha expressions
- **MultiIndex panel**: (symbol, timestamp) for clean time-series/cross-section separation
- **Free data only**: Binance Vision public klines, no paid data needed
- **42 transform functions**: ts_mean, ts_zscore, cs_rank, decay_linear, etc.
- **Deduplication**: SHA256 hash on expression string to avoid duplicates

## Disclaimer

This is an experimental research project. Use at your own risk. Past backtest performance does not guarantee future results. Start with small capital.

---

# 한국어

## 개요

GPT/Claude를 활용해 퀀트 알파를 자동 발굴하고, 백테스트로 검증한 뒤, Hyperliquid에서 달러 뉴트럴 포트폴리오를 자동 운용하는 시스템입니다.

### 작동 방식

```
1. 알파 발굴     : LLM이 트레이딩 전략 아이디어 생성 → 수학 수식으로 변환
2. 백테스트      : 과거 데이터로 수익성 검증 (Sharpe >= 0.5 필터)
3. 시그널 생성   : 검증된 알파 조합 → 일일 매매 시그널 생성
4. 자동 실행     : Hyperliquid에서 자동 리밸런싱 (cron)
```

### 전략

- **달러 뉴트럴**: 롱 비중 = 숏 비중 (시장 방향과 무관하게 수익 추구)
- **횡단면 분석**: 코인 간 상대 강도로 롱/숏 결정
- **멀티 알파**: 여러 알파를 Sharpe 비례 가중평균으로 조합
- **일일 리밸런싱**: 매일 오전 9시(한국시간) 자동 리밸런싱

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
# .env 파일에 API 키 입력
```

**필수 키:**
- `HYPERLIQUID_ACCOUNT_ADDRESS`: Hyperliquid 지갑 주소
- `HYPERLIQUID_SECRET_KEY`: Hyperliquid Agent Wallet 비밀키

**선택 키 (알파 발굴 시에만 필요):**
- `OPENAI_API_KEY`: OpenAI API 키

### 3. 바로 시작하기 (내장 알파 사용)

레포에 검증된 9개 알파가 포함되어 있습니다 (Sharpe >= 0.5, 2022~2026 데이터 백테스트 완료). 데이터 다운로드나 알파 발굴 없이 바로 트레이딩할 수 있습니다:

```bash
python main.py report              # 포함된 알파 확인
python main.py signal              # 오늘의 시그널 생성
python main.py execute             # 드라이런 (주문 미리보기)
python main.py execute --confirm   # Hyperliquid에서 실제 실행
```

### 4. (선택) 나만의 알파 발굴

추가 알파를 찾고 싶으면 OpenAI API 키가 필요합니다:

```bash
python main.py download                  # Binance 선물 캔들 데이터 다운로드
python main.py process                   # CSV → parquet 변환
python main.py pipeline --iterations 10  # ~1시간 소요, GPT 사용
```

### 5. 자본금 설정

`config/config.yaml`에서 자본금을 설정합니다:

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

**예시**: Hyperliquid에 $1,000이 있다면, `capital: 1500~2000`으로 설정 (2배 레버리지 기준 마진 사용률 75~100%)

레버리지 설정은 `src/execution/hyperliquid.py`에서 변경:
```python
COIN_LEVERAGE = {
    'BTC': 3,   # BTC, ETH: 3배 (변동성 낮음)
    'ETH': 3,
}
DEFAULT_LEVERAGE = 2  # 나머지: 2배
```

### 6. 시그널 생성 및 실행

```bash
python main.py signal              # 오늘의 시그널 확인
python main.py execute             # 드라이런 (주문 미리보기)
python main.py execute --confirm   # 실제 주문 실행
python main.py positions           # 현재 포지션 확인
```

### 7. 자동화 (cron)

```bash
# 매일 오전 9시 (한국시간) 자동 리밸런싱
crontab -e
0 0 * * * cd /path/to/ai-quantagent && python3 main.py execute --confirm >> logs/rebalance.log 2>&1
```

## 명령어 목록

| 명령어 | 설명 |
|--------|------|
| `download` | Binance 과거 캔들 데이터 다운로드 |
| `process` | CSV를 parquet으로 가공 |
| `pipeline` | 알파 발굴 파이프라인 실행 (LLM 사용) |
| `backtest` | 단일 알파 수식 백테스트 |
| `combine` | 상위 알파를 포트폴리오로 조합 |
| `signal` | 라이브 트레이딩 시그널 생성 |
| `execute` | 시그널 생성 후 Hyperliquid에서 실행 |
| `positions` | 현재 Hyperliquid 포지션 확인 |
| `report` | DB에서 상위 알파 조회 |

## 수익 구조

이 봇은 **시장 방향과 무관하게** 수익을 추구합니다:

- 강할 것으로 예측된 코인은 **롱** (매수)
- 약할 것으로 예측된 코인은 **숏** (공매도)
- 롱 = 숏이므로 시장이 오르든 내리든, **코인 간 상대적 차이**로 수익

```
시장 상승 시: 롱 코인이 더 많이 오르면 → 수익
시장 하락 시: 숏 코인이 더 많이 떨어지면 → 수익
손실 조건:   롱 코인이 숏 코인보다 못할 때
```

## 주의사항

이 프로젝트는 실험적 연구 목적입니다. 투자 손실에 대한 책임은 본인에게 있습니다. 과거 백테스트 성과가 미래 수익을 보장하지 않습니다. 소액으로 시작하세요.

## License

MIT
