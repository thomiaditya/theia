# This is the entry point for the command line script.

from pyexpat import model
from theia import Model


def main():
    print("Hello World!")
    model = Model()
    model.train()


# Dont bother below script.
if __name__ == "__main__":
    main()

# Stop right here. You dont need to exceed this line.
