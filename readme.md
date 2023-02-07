## Simple convenience package for OpenAI's API
A simple wrapper around OpenAI's python package, designed for my personal use. Will be updated sporadically.

Why make this? 

I previously was using openai.embeddings.util, however that package has a lot of bloat. I needed a slimmer package to use on pythonanywhere, so this was the solution. I also modified their version of caching (using a pickle)
to use jsonl instead and be faster (they load their cache a lot). I also made modifications to account for not having an unlimited rate limit.
