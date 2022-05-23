# This is the entry point for the command line script.
import sys
import argparse

sys.dont_write_bytecode = True

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument(
    "train", help="Train the already configured model. (You can configure the model in theia/config)")


def training():
    from theia import RetrievalModel

    model = RetrievalModel()
    model.fit()
    model.save()

    # model.load()
    # model.recommend("42")


def main():
    # Execute the parser.
    args = parser.parse_args()

    # Execute the command.
    if args.train:
        training()
        sys.exit(0)


# Dont bother below script.
if __name__ == "__main__":
    main()

# Stop right here. You dont need to exceed this line.
