"""
Jacksonville sentiment analysis using MIT TSGI archived data.

This script expects one or more county-level CSV files downloaded from
https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3IL00Q

Example:
  python jax_sentiment_analysis.py --files TSGI_county_2023.csv --outdir outputs
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

REQUIRED_COLUMNS = {
    "DATE",
    "NAME_0",
    "NAME_1",
    "NAME_2",
    "GID_2",
    "sentiment",
    "tweet_count",
}


@dataclass(frozen=True)
class GeoIdentifiers:
    name_0: str = "United States"
    name_1: str = "Florida"
    name_2: str = "Duval"
    gid_2: str = "USA.10.31_1"


class TSGIDataDownloader:
    """Load pre-processed sentiment data from MIT TSGI CSV exports."""

    def __init__(self, geo: GeoIdentifiers | None = None) -> None:
        self.geo = geo or GeoIdentifiers()

    def load_local_data(self, filepath: Path) -> pd.DataFrame:
        df = pd.read_csv(filepath)
        missing = REQUIRED_COLUMNS - set(df.columns)
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Missing required columns in {filepath}: {missing_list}")
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
        if df["DATE"].isna().any():
            raise ValueError(f"Invalid DATE values found in {filepath}")
        return df

    def load_multiple(self, filepaths: list[Path]) -> pd.DataFrame:
        frames = [self.load_local_data(path) for path in filepaths]
        combined = pd.concat(frames, ignore_index=True)
        return combined

    def filter_jacksonville(self, df: pd.DataFrame) -> pd.DataFrame:
        mask = (df["NAME_1"] == self.geo.name_1) & (df["NAME_2"] == self.geo.name_2)
        jax_data = df.loc[mask].copy()
        if jax_data.empty:
            raise ValueError("No Jacksonville/Duval County rows found in the provided data.")
        return jax_data


class JacksonvilleAnalyzer:
    """Analyze Jacksonville sentiment trends from TSGI data."""

    def __init__(self, data: pd.DataFrame) -> None:
        self.data = data.sort_values("DATE").reset_index(drop=True)

    def calculate_statistics(self) -> dict[str, float | str]:
        return {
            "mean_sentiment": self.data["sentiment"].mean(),
            "median_sentiment": self.data["sentiment"].median(),
            "std_sentiment": self.data["sentiment"].std(),
            "min_sentiment": self.data["sentiment"].min(),
            "max_sentiment": self.data["sentiment"].max(),
            "total_tweets": self.data["tweet_count"].sum(),
            "avg_daily_tweets": self.data["tweet_count"].mean(),
            "date_range": f"{self.data['DATE'].min()} to {self.data['DATE'].max()}",
        }

    def get_monthly_trends(self) -> pd.DataFrame:
        monthly = self.data.copy()
        monthly["year_month"] = monthly["DATE"].dt.to_period("M")
        monthly_agg = (
            monthly.groupby("year_month")
            .agg({"sentiment": ["mean", "std"], "tweet_count": "sum"})
            .reset_index()
        )
        monthly_agg.columns = [
            "year_month",
            "sentiment_mean",
            "sentiment_std",
            "total_tweets",
        ]
        monthly_agg["year_month"] = monthly_agg["year_month"].dt.to_timestamp()
        return monthly_agg

    def detect_significant_events(self, threshold_std: float = 2.0) -> pd.DataFrame:
        mean_sentiment = self.data["sentiment"].mean()
        std_sentiment = self.data["sentiment"].std()
        if std_sentiment == 0 or pd.isna(std_sentiment):
            return pd.DataFrame(columns=["DATE", "sentiment", "tweet_count", "z_score", "event_type"])

        data = self.data.copy()
        data["z_score"] = (data["sentiment"] - mean_sentiment) / std_sentiment
        significant_days = data.loc[abs(data["z_score"]) > threshold_std, [
            "DATE",
            "sentiment",
            "tweet_count",
            "z_score",
        ]].copy()
        if significant_days.empty:
            return significant_days.assign(event_type=pd.Series(dtype="object"))

        significant_days["event_type"] = significant_days["z_score"].apply(
            lambda x: "Positive spike" if x > 0 else "Negative dip"
        )
        return significant_days.sort_values("DATE")

    def seasonal_analysis(self) -> pd.DataFrame:
        seasons = {
            1: "Winter",
            2: "Winter",
            3: "Spring",
            4: "Spring",
            5: "Spring",
            6: "Summer",
            7: "Summer",
            8: "Summer",
            9: "Fall",
            10: "Fall",
            11: "Fall",
            12: "Winter",
        }
        seasonal = self.data.copy()
        seasonal["season"] = seasonal["DATE"].dt.month.map(seasons)
        seasonal_stats = (
            seasonal.groupby("season")
            .agg({"sentiment": ["mean", "std", "count"]})
            .reset_index()
        )
        seasonal_stats.columns = ["season", "sentiment_mean", "sentiment_std", "days"]
        return seasonal_stats


class JacksonvilleVisualizer:
    """Create visualizations for Jacksonville sentiment analysis."""

    def __init__(self) -> None:
        sns.set_style("whitegrid")
        self.jax_color = "#006747"

    def _finalize_plot(self, save_path: Path, show: bool) -> None:
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()

    def plot_daily_sentiment(self, data: pd.DataFrame, save_path: Path, show: bool) -> None:
        plot_data = data.copy()
        plot_data["sentiment_ma7"] = plot_data["sentiment"].rolling(window=7, center=True).mean()
        plot_data["sentiment_ma30"] = plot_data["sentiment"].rolling(window=30, center=True).mean()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        ax1.scatter(plot_data["DATE"], plot_data["sentiment"], alpha=0.2, s=10, color="gray", label="Daily")
        ax1.plot(plot_data["DATE"], plot_data["sentiment_ma7"], color=self.jax_color, linewidth=2, label="7-day average")
        ax1.plot(plot_data["DATE"], plot_data["sentiment_ma30"], color="#D4AA00", linewidth=2, linestyle="--", label="30-day average")
        ax1.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="Neutral (0.5)")
        ax1.set_ylabel("Sentiment Score (0-1)", fontsize=12, fontweight="bold")
        ax1.set_title("Jacksonville, FL - Daily Sentiment Trend (MIT TSGI Data)", fontsize=14, fontweight="bold")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        ax2.bar(plot_data["DATE"], plot_data["tweet_count"], color=self.jax_color, alpha=0.6)
        ax2.set_ylabel("Daily Tweet Count", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Date", fontsize=12, fontweight="bold")
        ax2.grid(alpha=0.3)

        self._finalize_plot(save_path, show)

    def plot_monthly_trends(self, monthly_data: pd.DataFrame, save_path: Path, show: bool) -> None:
        fig, ax = plt.subplots(figsize=(14, 6))

        ax.plot(
            monthly_data["year_month"],
            monthly_data["sentiment_mean"],
            marker="o",
            linewidth=2.5,
            color=self.jax_color,
            markersize=8,
        )
        ax.fill_between(
            monthly_data["year_month"],
            monthly_data["sentiment_mean"] - monthly_data["sentiment_std"],
            monthly_data["sentiment_mean"] + monthly_data["sentiment_std"],
            alpha=0.2,
            color=self.jax_color,
        )
        ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.5)
        ax.set_xlabel("Month", fontsize=12, fontweight="bold")
        ax.set_ylabel("Average Sentiment Score", fontsize=12, fontweight="bold")
        ax.set_title("Jacksonville Monthly Sentiment Trends", fontsize=14, fontweight="bold")
        ax.grid(alpha=0.3)
        plt.xticks(rotation=45)

        self._finalize_plot(save_path, show)

    def plot_seasonal_comparison(self, seasonal_stats: pd.DataFrame, save_path: Path, show: bool) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))

        seasons_order = ["Winter", "Spring", "Summer", "Fall"]
        seasonal_stats = seasonal_stats.set_index("season").reindex(seasons_order).reset_index()

        ax.bar(
            seasonal_stats["season"],
            seasonal_stats["sentiment_mean"],
            color=[self.jax_color, "#D4AA00", "#006747", "#9E7C0C"],
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )
        ax.errorbar(
            seasonal_stats["season"],
            seasonal_stats["sentiment_mean"],
            yerr=seasonal_stats["sentiment_std"],
            fmt="none",
            color="black",
            capsize=5,
            capthick=2,
        )
        ax.axhline(y=0.5, color="red", linestyle=":", alpha=0.5)
        ax.set_ylabel("Average Sentiment Score", fontsize=12, fontweight="bold")
        ax.set_title("Jacksonville Sentiment by Season", fontsize=14, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

        self._finalize_plot(save_path, show)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Jacksonville sentiment using MIT TSGI data.")
    parser.add_argument(
        "--files",
        nargs="+",
        required=True,
        help="One or more county-level TSGI CSV files.",
    )
    parser.add_argument(
        "--outdir",
        default="outputs",
        help="Directory to store generated PNG files.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Z-score threshold for significant sentiment events.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots in a window (default: save only).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    filepaths = [Path(path) for path in args.files]
    missing = [path for path in filepaths if not path.exists()]
    if missing:
        missing_list = ", ".join(str(path) for path in missing)
        raise SystemExit(f"Missing input files: {missing_list}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    downloader = TSGIDataDownloader()
    all_data = downloader.load_multiple(filepaths)
    jax_data = downloader.filter_jacksonville(all_data)

    analyzer = JacksonvilleAnalyzer(jax_data)
    stats = analyzer.calculate_statistics()

    print("JACKSONVILLE SENTIMENT SUMMARY")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")

    monthly = analyzer.get_monthly_trends()
    significant = analyzer.detect_significant_events(threshold_std=args.threshold)
    seasonal = analyzer.seasonal_analysis()

    if not significant.empty:
        print("\nSignificant sentiment days:")
        print(significant.to_string(index=False))
    else:
        print("\nNo significant sentiment spikes detected.")

    visualizer = JacksonvilleVisualizer()
    visualizer.plot_daily_sentiment(jax_data, outdir / "jax_daily_sentiment.png", args.show)
    visualizer.plot_monthly_trends(monthly, outdir / "jax_monthly_trends.png", args.show)
    visualizer.plot_seasonal_comparison(seasonal, outdir / "jax_seasonal.png", args.show)


if __name__ == "__main__":
    main()
