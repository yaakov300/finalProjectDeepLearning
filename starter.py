import yaml

def main ():
    config = yaml.safe_load(open("config.yml"))


if __name__ == "__main__":
    main()
