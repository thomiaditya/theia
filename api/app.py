# This is the entry point for the command line script.
import sys
import argparse

# Parse the command line arguments.
parser = argparse.ArgumentParser()
parser.add_argument("train", help="Train the model.")


def training():
    from theia import Model

    model = Model()
    model.train()


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
