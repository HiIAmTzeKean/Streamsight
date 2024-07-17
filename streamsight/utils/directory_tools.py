import os


def safe_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_config_yaml(filename):
    content = """\
version: 1
formatters:
  colored:
    (): colorlog.ColoredFormatter
    format: '%(log_color)s%(levelname)-8s%(reset)s - %(name)s - %(blue)s%(message)s'
  simple:
    format: '%(asctime)s: - %(name)s - %(levelname)s - %(message)s'
    datefmt: "%Y-%m-%d %H:%M:%S"
    class: logging.Formatter
handlers:
  console:
    class: logging.StreamHandler
    level: DEBUG
    formatter: colored
    stream: ext://sys.stdout
  file:
    class: logging.FileHandler
    level: DEBUG
    formatter: simple
    filename: ./log/output.log
loggers:
  root:
    level: DEBUG
    handlers: [console,file]
"""
    with open(filename, "w") as f:
        f.write(content)
