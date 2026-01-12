# 🧠🐭😥 social-defeat
Active inference modeling of social defeat in mice

I really need to clean up this repo
Check out *het_fit.ipynb* and *sim.py* for running sims & visualizing, ignore the rest 

## 🚀 Project Setup

Follow these steps to set up the project and its dependencies on a new machine. This project uses a custom version of `pymdp` included as a submodule.

### 1\. Clone the Repository

Clone this repository using the `--recurse-submodules` flag to ensure the `pymdp` submodule is downloaded as well.

```bash
git clone --recurse-submodules https://github.com/hridaik/social-defeat
cd social-defeat
```

### 2\. Create and Activate a Virtual Environment

It's crucial to use a virtual environment to manage project-specific dependencies.

  * **On macOS / Linux:**

    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

  * **On Windows:**

    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

### 3\. Install Dependencies

First, install the local `pymdp` submodule in "editable" mode. Then, install the rest of the required packages from `requirements.txt`.

```bash
# Install the local pymdp package
pip install -e pymdp/

# Install all other packages
pip install -r requirements.txt
```

You are now ready to run the project\! ✅

