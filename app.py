from pprint import pprint
import utils
import training
import testing
import sys
from config import Configuration


if __name__ == '__main__':

    # Load configuration from file or arguments
    config = Configuration()

    print("\nRun torcs env with this configuration?\n")
    pprint(config.args)

    # Create basic directories if not exists
    utils.mkdir_resources(config.args['resources'])

    while True:
        question = input("\n\nPlease, press one key:\n 1. TRAIN\n 2. TEST\n 0. EXIT\n")
        question = int(question)
        if question == 1:
            utils.to_json(config.args)
            training.run()
        elif question == 2:
            testing.run()
        elif question == 0:
            sys.exit(0)
        else:
            continue
