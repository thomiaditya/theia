# This is the entry point for the command line script.
import os
import sys
import argparse


def main(cmd=None):
    # Setting up the argument parser
    parser = argparse.ArgumentParser(
        description="Recommendation Engine for Mental Health Apps")
    parser._positionals.title = "commands"

    subparsers = parser.add_subparsers(dest="command")

    # Subparser for the train command
    train = subparsers.add_parser(
        "train", help="Train the model with spesific settings")
    train.add_argument("-e", "--epochs", type=int, default=None,
                       help="Number of epochs")

    # Subparser for the server command
    server = subparsers.add_parser(
        "server", help="Server using FastAPI")

    # Create subparser for server command
    server_subparser = server.add_subparsers(dest="server_command")
    server_subparser.add_parser("start", help="Start the server")

    # Subparser for the predict command
    predict = subparsers.add_parser(
        "predict", help="Predict ratings for a given user")
    predict.add_argument("-u", "--user", type=int, required=True,
                         help="User id")

    # Parse the arguments
    args = parser.parse_args(cmd)

    # Run the command
    if args.command == "train":
        from theia import RetrievalModel
        model = RetrievalModel(epochs=args.epochs)
        model.train()
        model.save()
        sys.exit(0)
    
    if args.command == "server":
        pass
    
    if args.command == "predict":
        from theia import RetrievalModel
        model = RetrievalModel()
        print(model.recommend(str(args.user), "last_saved"))
        sys.exit(0)

    # If no command is specified, print the help
    parser.print_help()

    # Dont bother below script.
if __name__ == "__main__":
    main()

# Stop right here. You dont need to exceed this line.
