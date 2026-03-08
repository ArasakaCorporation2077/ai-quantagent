# AI Quantagent

Public Hyperliquid runner for a dollar-neutral crypto futures strategy.

[English](#overview) | [한국어](#개요)

This repo is meant for running and testing the live strategy, not for exposing the full internal research pipeline. It ships with a curated alpha database and the runtime needed to generate signals and rebalance positions.

## Overview

Default live mode runs a daily `1d` rebalance using OOS-first alpha selection:

- rank alphas by `sharpe_oos`
- require `n_oos` history before live use
- prune correlated alphas
- combine them into a dollar-neutral portfolio

The private LLM alpha-generation workflow is not included in this public repo.

## Strategy

- Dollar neutral: long exposure equals short exposure
- Cross-sectional: long stronger coins, short weaker coins
- OOS-first selection: live ranking defaults to `sharpe_oos >= 0.5` and `n_oos >= 60`
- Daily rebalance: default live frequency is `1d` at UTC 00:00 / KST 09:00

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

Required keys:

- `HYPERLIQUID_ACCOUNT_ADDRESS`
- `HYPERLIQUID_SECRET_KEY`

### 3. Set Capital

Edit `config/config.yaml`:

```yaml
execution:
  capital: 800
```

If `execution.capital_multiplier` is greater than `0`, the bot uses account value times that multiplier for live execution.

### 4. Run

```bash
python main.py report              # Top OOS-ranked alphas from the shipped DB
python main.py signal              # Live signal (default: 1d)
python main.py execute             # Dry run
python main.py execute --confirm   # Live execution
python main.py positions           # Current Hyperliquid positions
```

`signal` and `execute` are the main public-runner commands. Research commands such as `combine`, `half-life`, `backtest`, `download`, and `process` are for local analysis.

### 5. Automate

```bash
# Daily rebalance at UTC 00:00
crontab -e
0 0 * * * cd /path/to/ai-quantagent && python3 main.py execute --confirm >> logs/rebalance.log 2>&1
```

## Commands

| Command | Description |
|---------|-------------|
| `signal` | Generate live trading signal from OOS-selected alphas |
| `execute` | Generate signal and execute on Hyperliquid |
| `positions` | Show current Hyperliquid positions |
| `report` | Show top alphas from the shipped DB |
| `combine` | Combine top alphas into a portfolio and compare methods |
| `half-life` | Analyze signal decay and rebalance persistence |
| `backfill-oos` | Recompute missing OOS metrics for older DB rows |
| `backtest` | Backtest a single alpha expression |
| `download` | Download historical kline data from Binance |
| `process` | Process raw CSVs into enriched parquet files |

## Data Notes

- The repo ships with a SQLite alpha database at `db/alpha_research.db`.
- Live `signal` and `execute` use the shipped DB plus recent Binance candles.
- Research commands are better with local historical parquet data.

## Architecture

```text
src/
|- alpha/          # Alpha parser, evaluator, validator, transforms
|- backtest/       # Backtesting engine, metrics, position sizing
|- data/           # Binance data downloader and processor
|- execution/      # Hyperliquid order execution
|- orchestrator/   # Alpha combiner and live signal generator
|- storage/        # SQLite alpha and backtest storage
`- config.py       # Configuration loader
```

## Disclaimer

This is an experimental research project. Use at your own risk. Past backtest performance does not guarantee future results.

---

## 개요

Hyperliquid에서 실행하는 달러 뉴트럴 크립토 선물 전략용 공개 실행 저장소입니다.

이 저장소는 내부 연구 파이프라인 전체를 공개하기 위한 것이 아니라, 선별된 알파 DB와 실거래 실행 스택을 배포해서 다른 사람이 직접 시그널 생성과 리밸런싱을 테스트할 수 있게 하는 용도입니다.

기본 실거래 모드는 `1d` 일일 리밸런싱이며, 알파 선택은 OOS 우선 기준을 사용합니다.

- `sharpe_oos` 기준으로 알파 랭킹
- `n_oos` 표본 수가 충분한 알파만 사용
- 상관 높은 알파 제거
- 달러 뉴트럴 포트폴리오로 결합

비공개 LLM 알파 생성 파이프라인은 이 공개 저장소에 포함되어 있지 않습니다.

## 전략

- 달러 뉴트럴: 롱 익스포저와 숏 익스포저가 같음
- 횡단면 전략: 상대적으로 강한 코인을 롱, 약한 코인을 숏
- OOS 우선 선별: 기본 라이브 랭킹은 `sharpe_oos >= 0.5`, `n_oos >= 60`
- 일일 리밸런싱: 기본 라이브 주기는 `1d`이며 UTC 00:00 / KST 09:00 기준

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
# .env에 Hyperliquid 키 입력
```

필수 키:

- `HYPERLIQUID_ACCOUNT_ADDRESS`
- `HYPERLIQUID_SECRET_KEY`

### 3. 자본 설정

`config/config.yaml` 수정:

```yaml
execution:
  capital: 800
```

`execution.capital_multiplier`가 `0`보다 크면, 실거래 시 계좌 평가금액에 해당 배수를 곱한 값을 자동 자본으로 사용합니다.

### 4. 실행

```bash
python main.py report              # 배포된 DB 기준 상위 OOS 알파 확인
python main.py signal              # 라이브 시그널 확인 (기본: 1d)
python main.py execute             # 드라이런
python main.py execute --confirm   # 실제 주문 실행
python main.py positions           # 현재 Hyperliquid 포지션 확인
```

`signal`과 `execute`가 공개 실행 저장소의 핵심 명령입니다. `combine`, `half-life`, `backtest`, `download`, `process`는 로컬 연구/분석용 명령입니다.

### 5. 자동화

```bash
# UTC 00:00 일일 리밸런싱
crontab -e
0 0 * * * cd /path/to/ai-quantagent && python3 main.py execute --confirm >> logs/rebalance.log 2>&1
```

## 명령어

| 명령어 | 설명 |
|--------|------|
| `signal` | OOS 기준으로 선별된 알파로 라이브 시그널 생성 |
| `execute` | 시그널 생성 후 Hyperliquid에 실행 |
| `positions` | 현재 Hyperliquid 포지션 확인 |
| `report` | 배포된 DB 기준 상위 알파 조회 |
| `combine` | 상위 알파를 결합해서 성능 비교 |
| `half-life` | 시그널 감쇠와 리밸런싱 지속성 분석 |
| `backfill-oos` | 오래된 DB 행의 OOS 지표 재계산 |
| `backtest` | 단일 알파 수식 백테스트 |
| `download` | Binance 과거 캔들 다운로드 |
| `process` | CSV를 parquet로 가공 |

## 데이터 메모

- 저장소에는 `db/alpha_research.db` SQLite 알파 DB가 포함됩니다.
- 라이브 `signal`, `execute`는 배포된 DB와 최신 Binance 캔들을 함께 사용합니다.
- 연구용 명령은 로컬 과거 parquet 데이터가 있을 때 더 잘 동작합니다.

## 구조

```text
src/
|- alpha/          # 알파 파서, 평가기, 검증기, 변환 함수
|- backtest/       # 백테스트 엔진, 지표 계산, 포지션 사이징
|- data/           # Binance 데이터 다운로드 및 가공
|- execution/      # Hyperliquid 주문 실행
|- orchestrator/   # 알파 결합 및 라이브 시그널 생성
|- storage/        # SQLite 알파 / 백테스트 저장소
`- config.py       # 설정 로더
```

## 주의

이 프로젝트는 실험적 연구 프로젝트입니다. 모든 사용 책임은 본인에게 있으며, 과거 백테스트 성과는 미래 수익을 보장하지 않습니다.
