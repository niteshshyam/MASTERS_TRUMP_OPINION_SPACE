# MASTERS_TRUMP_OPINION_SPACE
Repository containing the working files of the Trump Opinion Space masters research

## Files
### oil_posts_v3.json
the 198 oil post corpus used in the scoring and subsequent tests, extracted using the two-stage filtering
### oil_rubric_v1_4.json 
rubric version 1.4 containing 11 features, eligibility rating, and exclusion reasoning. Version 1.3 was ran by the LLMs, version 1.4 just includes the "scores" key to explicitly show what scores can be assigned. 
### oil_v3.ipynb
the working Jupyter notebook for carrying out all the steps in the methodology
### scoring_3LLM_v4.py 
scoring script for the Open Source Ollama models using version 1.4 rubric. Runs Llama, Mistral, and Qwen consecutively three times.
### url_archive.json
posts that contained URLs are resolved here to their headline/subheading or article excerpt to then be used in the scoring
