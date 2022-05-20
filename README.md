# Theia

Theia is a recommender system for therapists that considers both the therapist's and the client's preferences to make recommendations.
The therapist's preferences are considered when making suggestions for clients. The therapist can set different weights for different criteria, such as the type of therapy, the client's needs, and the therapist's expertise.
The client's preferences are considered when making suggestions for therapists. The client can set different weights for different criteria, such as the type of therapy, the therapist's approach, and the therapist's location.
Theia takes into account both the therapist's and the client's preferences to make the best possible recommendations.

<!--
Theia is easy to use - simply install the software and point your camera at a person's face. Theia will analyze the person's facial expressions and voice to generate a sentiment score. The score will range from -1 (very negative) to 1 (very positive), with 0 being neutral. Theia can also generate a report that includes a breakdown of the person's sentiment by different emotions.

Theia is accurate and reliable, and has been validated against ground truth data. Theia is also private and secure, and does not store any personally identifiable information.

Theia can be used by businesses to understand customer sentiment, or by individuals to better understand their own emotions. Theia is the perfect tool for anyone who wants to better understand their emotions, or the emotions of others. -->

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

- [Python](https://www.python.org/downloads/), version 3.6 or later
- [Python Virtual Environment](https://virtualenv.pypa.io/en/stable/), version 16.0 or later

### Installing

A step by step series of examples that tell you how to get a development env running

Clone the repository using below command

```bash
git clone https://github.com/thomiaditya/theia.git
cd theia
```

#### Windows

```powershell
# Activate the virtual environment
python -m venv venv
.\venv\Scripts\activate
pip install .
```

#### Linux

```bash
# Activate the virtual environment
python3 -m venv venv
source venv/bin/activate
pip3 install .
```

After that you can run the application using below command

```bash
theia-cli # Still under development
```

Above command will run the application in the current directory.

## Running the tests

Explain how to run the automated tests for this system

### Break down into end to end tests

Explain what these tests test and why

```

Give an example

```

### And coding style tests

Explain what these tests test and why

```

Give an example

```

## Deployment

Add additional notes about how to deploy this on a live system

## Built With

- [Tensorflow](https://www.tensorflow.org/), for model training and deployment also [Keras](https://keras.io/)
- [Wandb](https://wandb.ai), for hyperparameter tuning and experiment tracking
- [Python Virtual Environment](https://docs.python.org/3/tutorial/venv.html), to manage project dependencies

## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/PurpleBooth/b24679402957c63ec426) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags).

## Authors

- **Thomi Aditya** - _Initial work_ - [Github](https://github.com/thomiaditya)

See also the list of [contributors](https://github.com/thomiaditya/theia/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

- Hat tip to anyone whose code was used
- Inspiration
- etc

```

```
