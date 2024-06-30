## Similarity Recommender

A script that takes a list of paths traversed by a website's visitor and, given historical data about how other past visitors have browsed the site, attempts to recommend a page that the current visitor has not yet seen.

For example, if historically, many visitors visit pages A, B, C, and D, then it's likely that a current user, having already visited pages A, B, and D, might also be interested in seeing page C.

The code works by using numpy to generate a vector whose dimensions scale linearly with the number of unique paths and metadata keys that are available in the historical data. Each element in the vector represents one unique path or metadata key and has a value of either 0 or 1; 1 indicates that a visitor has visited that page.

It then uses [cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) to find the vector that most closely matches the current path (`new_path`) taken by the current visitor.


### Run this code

1. Clone this repository and `cd` into it
2. Create a virtualenv: `python3 -m venv venv`
3. Enter the virtualenv: `source venv/bin/activate`
4. Install requirements: `pip3 install -r requirements.txt`
5. Run the script: `python3 similarity-recommender.py`