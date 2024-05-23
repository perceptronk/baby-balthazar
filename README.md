# Baby Balthazar
A very simple recommendation engine that recommends similar content based on description text similarity
Based on https://www.kaggle.com/code/aishwaryasharma1992/recommender-system-using-huggingface-library

## Getting started
Create a virtual environment and install dependencies
```
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Start the application with
```
python main.py
```

You can then access the server at http://localhost:8080/docs

You can send a test request using curl with
```
curl -XPOST --header 'Content-Type: application/json' --data @example_input.json http://localhost:8080/recommend
```

