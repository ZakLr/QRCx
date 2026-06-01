def main():
    from .pipeline import QRCPipeline
    from .config import ExperimentConfig
    import argparse

    parser = argparse.ArgumentParser(description="CUDA-Q QRC Weather Forecasting")
    parser.add_argument("--config", type=str, help="Path to YAML config file")
    parser.add_argument("--years", type=int, nargs=2, default=[2019, 2024],
                        help="Start and end year for data download")
    args = parser.parse_args()

    cfg = ExperimentConfig.from_yaml(args.config) if args.config else ExperimentConfig()
    pipeline = QRCPipeline(config=cfg)
    df = pipeline.load_data(args.years[0], args.years[1])
    pipeline.preprocess(df)
    results = pipeline.run()
    results.plot_all()
    print("Done — figures saved to", cfg.figures_dir)


if __name__ == "__main__":
    main()
