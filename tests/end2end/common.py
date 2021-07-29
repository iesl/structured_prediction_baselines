from pathlib import Path

config_directory = (Path(__file__).parent / "configs").absolute()
vocab_directory = (
    Path(__file__).parent / "assets" / "bibtex_vocabulary"
).absolute()


root_directory = (Path(__file__).parent / "../../").absolute()
configs = sorted(list(config_directory.glob("*.json")))
marks = [f"{config.stem}: {config.stem} " for config in configs]
