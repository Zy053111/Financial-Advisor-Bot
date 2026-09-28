# Financial Advisor Bot (FAB)

An institutional-grade, Explainable Deep Reinforcement Learning (DRL) Portfolio Management System for the Magnificent 7 equities basket. The platform integrates Proximal Policy Optimization (PPO), out-of-sample backtesting with market friction models, and local LLM-driven Explainable AI (XAI).

---

## Architecture Overview

* **DRL Engine**: PPO (Continuous Actor-Critic via Stable-Baselines3) managing continuous portfolio weights on the simplex.
* **Feature Engineering**: Multi-asset Log Returns, 14-day RSI, and MACD (26, 12, 9).
* **Backtesting**: Backtrader simulation engine featuring 0.10% commission fees, 0.05% execution slippage, and rolling Markowitz MPT / 1/N equal-weight benchmarks.
* **XAI Dialogue**: Local Llama-3 via Ollama generating persona-aligned strategy narratives and telemetry-grounded Q&A.
* **Frontend**: React, Tailwind CSS, Lucide Icons, and Recharts interactive visualizations.

---

## Prerequisites

Ensure you have the following installed on your system:
* Python 3.10+
* Node.js 18+ & npm
* [Ollama](https://ollama.com/download)

---

## Installation & Environment Setup

### 1. Local LLM Setup (Ollama)
Download and install Ollama from [ollama.com](https://ollama.com/download). Once installed, run the local Llama-3 model in your terminal:

    ollama run llama3

Keep this service running in the background at http://localhost:11434.

### 2. Backend Environment & Dependencies
Open a separate terminal in the project root directory, create a Python virtual environment, activate it, and install all required packages:

    # Create virtual environment
    python -m venv venv

    # Activate virtual environment
    # Windows (Command Prompt / PowerShell)
    venv\Scripts\activate
    # macOS / Linux
    source venv/bin/activate

    # Install dependencies from requirements.txt
    pip install -r requirements.txt

### 3. Frontend Setup (React)
Open a separate terminal and configure the frontend dependencies:

    cd frontend
    npm install

---

## Running Verification & Automated Unit Tests

Execute the automated test suite to verify mathematical simplex budget invariants, rolling Sharpe singularities, risk profile clamping, and fallback handlers:

    # Ensure virtual environment is active in the project root
    python backend/tests.py

---

## Running Model Training & Experimental Artifacts

Note: Pre-trained weights and pre-computed backtest histories are saved in backend/saved_models/ and backend/backtest_history.json. You only need to run these if retraining or reproducing thesis figures.

* Train Primary PPO Agent (In-Sample: 2020-01-01 to 2024-12-31):
    ```bash
    python backend/train_ppo.py
    ```

* Run Out-of-Sample Backtest (OOS: 2025-01-01 to 2026-08-30):
  Generates backtest_result.png (DRL vs. Markowitz vs. 1/N) and updates backtest_history.json.
    ```bash
    python backend/backtest.py
    ```

* Execute Thesis Ablation Study:
  Trains the ablated baseline (Returns-only) and exports ablation_result.png:
    ```bash
    python backend/ablation_study.py
    ```

---

## Starting the Application (Demo & Presentation)

Run the backend and frontend development servers concurrently:

### Terminal 1: Backend REST API

    # Activate virtual environment if not already active
    venv\Scripts\activate  # Windows
    # source venv/bin/activate  # macOS / Linux

    python manage.py runserver

The Django REST API runs on http://127.0.0.1:8000.

### Terminal 2: Frontend Dashboard

    cd frontend
    npm run dev

Open your browser and navigate to the local Vite URL (typically http://localhost:5173) to interact with the dashboard.

---

## Core Features Walkthrough

1. Mandate Selection: Toggle between Conservative, Moderate, and Aggressive profiles to observe dynamic weight transformations in the donut chart and table.
2. Execution Order Book: Adjust portfolio capital to inspect buy/sell deltas and estimated shares calculated against the equal-weighted 1/N baseline.
3. Out-of-Sample Trajectory: Inspect interactive equity curves comparing the PPO Agent, rolling Markowitz MPT, and the 1/N Benchmark under friction models.
4. XAI Dialogue & Suggestion Chips: Click pre-configured prompt chips or submit queries to chat with the Llama-3 institutional advisor grounded in live telemetry.
