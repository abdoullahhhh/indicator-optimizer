import numpy as np
import pandas as pd
import vectorbt as vbt
import optuna, json
from optuna.exceptions import TrialPruned

# Import your indicator from the same package
from .indicators import rsi_threshold_strategy

class SingleIndicatorStrategy:
    def __init__(self, params=None):
        # params: [rsi_len, pmom, nmom, ema_len]
        self.default_params = [14, 65, 32, 5]
        self.params = params or self.default_params

    def get_signals(self, data: pd.DataFrame) -> np.ndarray:
        rsi_len, pmom, nmom, ema_len = self.params
        regime = rsi_threshold_strategy(data, length=rsi_len, pmom=pmom, nmom=nmom, ema_len=ema_len)
        return regime.fillna(0).to_numpy()

    def backtest(self, data: pd.DataFrame, signals: np.ndarray, dates=None, plot=False, shorts=False):
        price = pd.Series(data['close'].astype(float), index=data.index)
        regime = pd.Series(signals, index=price.index)
        entries = (regime.diff().fillna(0) > 0)
        exits = (regime.diff().fillna(0) < 0)

        pf = vbt.Portfolio.from_signals(
            price,
            entries=entries,
            exits=exits,
            short_entries=exits if shorts else None,
            short_exits=entries if shorts else None,
            freq='1D'
        )
        if plot:
            fig = pf.plot()
            fig.update_yaxes(type='log', row=1)
            fig.show()
        return pf

    def clustered_trade_penalty(self, signals: np.ndarray) -> float:
        trade_indices = np.where(np.diff(signals) != 0)[0]
        if len(trade_indices) == 0:
            return 1.0
        trade_distances = np.diff(trade_indices)
        decay_penalty = np.exp(-0.3 * (trade_distances - 10))
        return 1 - np.sum(decay_penalty) / len(trade_indices)

    def long_hold_penalty(self, regime: pd.Series, max_days: int = 365, k: float = 1.0) -> float:
        """
        Penalize continuous positions that last longer than max_days.
        Returns a multiplicative factor in (0, 1], where 1 means no penalty.

        The penalty scales as exp(-k * excess_days / max_days) for each long run,
        multiplied across runs. With k=1, holding 2 years continuously gives ~e^(-1) ~ 0.37.
        """
        if regime.empty:
            return 1.0

        # consider non-zero regime only (in-position)
        in_pos = regime.replace({0: np.nan}).notna()

        # group consecutive runs
        run_id = (regime != regime.shift()).cumsum()
        penalties = []

        for rid, run in regime.groupby(run_id):
            if run.iloc[0] == 0:
                continue  # flat run
            run_days = (run.index[-1] - run.index[0]).days + 1
            if run_days > max_days:
                excess = run_days - max_days
                penalties.append(np.exp(-k * excess / max_days))

        if not penalties:
            return 1.0

        penalty = float(np.prod(penalties))
        return penalty


    def safe_normalize_returns(self, return_value: float, scale_factor: float = 100) -> float:
        x = np.clip(return_value, -10 * scale_factor, 10 * scale_factor) / scale_factor
        return 1 / (1 + np.exp(-x))

    def evaluate_score_configurable(self, pf, signals=None, scoring_config=None) -> float:
        stats = pf.stats()

        risk_metrics = (
            -abs(stats['Max Drawdown [%]']) / 100 +
            -abs(stats.get('Worst Trade [%]', 0)) / 100
        ) / 2

        risk_adjusted_returns = (
            stats.get('Sharpe Ratio', 0) / 5 +
            stats.get('Sortino Ratio', 0) / 7 +
            stats.get('Omega Ratio', 0) / 3
        ) / 3

        total_return = self.safe_normalize_returns(pf.total_return() * 100, scale_factor=scoring_config['normalization_scale']['total_return'])
        annualized_return = self.safe_normalize_returns(pf.annualized_return() * 100, scale_factor=scoring_config['normalization_scale']['annualized_return'])

        win_rate = stats.get('Win Rate [%]', 0) / 100
        profit_factor = min(stats.get('Profit Factor', 1.0), 10) / 10
        win_metrics = (win_rate + profit_factor) / 2

        weights = scoring_config['weights']
        base_score = (
            weights['risk_metrics'] * risk_metrics +
            weights['risk_adjusted_returns'] * risk_adjusted_returns +
            weights['win_metrics'] * win_metrics +
            weights['total_returns'] * total_return +
            weights['annualized_returns'] * annualized_return
        )

        if signals is not None:
            coherence = np.exp(self.clustered_trade_penalty(signals) - 1)
            base_score *= coherence

        return float(base_score)


    def get_scoring_presets(self):
        return {
            'high_risk': {
                'weights': {
                    'risk_metrics': 0.05,
                    'risk_adjusted_returns': 0.10,
                    'win_metrics': 0.15,
                    'total_returns': 0.45,
                    'annualized_returns': 0.25
                },
                'normalization_scale': {'total_return': 50, 'annualized_return': 100}
            },
            'balanced': {
                'weights': {
                    'risk_metrics': 0.15,
                    'risk_adjusted_returns': 0.25,
                    'win_metrics': 0.25,
                    'total_returns': 0.20,
                    'annualized_returns': 0.15
                },
                'normalization_scale': {'total_return': 50, 'annualized_return': 100}
            },
            'conservative': {
                'weights': {
                    'risk_metrics': 0.34,
                    'risk_adjusted_returns': 0.30,
                    'win_metrics': 0.20,
                    'total_returns': 0.10,
                    'annualized_returns': 0.05
                },
                'normalization_scale': {'total_return': 50, 'annualized_return': 100}
            }
        }


    def optimize_params(self, data: pd.DataFrame, n_trials: int = 200, scoring_config=None, min_trades: int = 5, preset: str = 'balanced'):   # 'high_risk' or 'conservative'
        """
        Optimize parameters with pruning and long-hold penalty.
        - Prunes trials with no/too few trades or NaN/inf stats
        - Penalizes continuous holds longer than 1 year
        - Uses tight parameter ranges
        - Shows progress bar and per-trial updates
        """


        if scoring_config is None:
             scoring_config = self.get_scoring_presets().get(preset, self.get_scoring_presets()['balanced'])

        # Pruner: discard weak trials early after some warmup trials
        pruner = optuna.pruners.MedianPruner(n_startup_trials=20, n_warmup_steps=20)
        study = optuna.create_study(direction='maximize', pruner=pruner)

        def objective(trial):
            # 3) Realistic parameter ranges
            params = [
                trial.suggest_int('rsi_len', 6, 35),
                trial.suggest_int('pmom', 50, 78),
                trial.suggest_int('nmom', 22, 50),
                trial.suggest_int('ema_len', 3, 15),
            ]
            self.params = params

            # Build signals and entries/exits
            signals = self.get_signals(data)
            regime = pd.Series(signals, index=data.index)
            entries = (regime.diff().fillna(0) > 0)
            exits = (regime.diff().fillna(0) < 0)
            trades = int(entries.sum())

            # 1) Prune “bad” trials early
            if trades == 0 or trades < min_trades:
                raise TrialPruned(f"Pruned: trades={trades} (< {min_trades})")

            # Backtest
            pf = self.backtest(data, signals, plot=False)
            stats = pf.stats()

            # NaN / inf checks
            critical_vals = [
                pf.total_return(), pf.annualized_return(),
                stats.get('Sharpe Ratio', np.nan), stats.get('Win Rate [%]', np.nan)
            ]
            if any((not np.isfinite(x)) or pd.isna(x) for x in critical_vals):
                raise TrialPruned("Pruned: NaN/inf in key metrics")

            # Base score
            score = self.evaluate_score_configurable(pf, signals=signals, scoring_config=scoring_config)

            # 2) Penalize “multi-year holds”
            hold_pen = self.long_hold_penalty(regime=regime, max_days=365, k=1.0)
            score *= hold_pen

            # Final sanity
            if not np.isfinite(score) or pd.isna(score):
                raise TrialPruned("Pruned: NaN/inf score")

            return float(score)

        # 4) Make progress visible
        def progress_cb(study, trial):
            status = "PRUNED" if trial.state.name == "PRUNED" else f"value={trial.value:.5f}" if trial.value is not None else "PENDING"
            print(f"Trial {trial.number + 1}/{n_trials}: {status} | params={trial.params}")

        study.optimize(
            objective,
            n_trials=n_trials,
            show_progress_bar=True,   
            callbacks=[progress_cb],
            n_jobs=1
        )

        best = study.best_params
        self.params = [best['rsi_len'], best['pmom'], best['nmom'], best['ema_len']]
        print(f"Best trial: #{study.best_trial.number} value={study.best_value:.6f}")
        return self.params

    def to_json(self):
        return json.dumps({"params": self.params})

    @classmethod
    def from_json(cls, json_str):
        data = json.loads(json_str)
        return cls(params=data["params"])