import json
import random
import anthropic
from openai import OpenAI


def calc_price(model, usage):
    if "claude" in model:
        return (0.015 * usage.input_tokens + 0.075 * usage.output_tokens) / 1000.0
    if model == "gpt-4-1106-preview" or model == "gpt-4-0125-preview":
        return (0.01 * usage.prompt_tokens + 0.03 * usage.completion_tokens) / 1000.0
    if model == "gpt-4":
        return (0.03 * usage.prompt_tokens + 0.06 * usage.completion_tokens) / 1000.0
    if (model == "gpt-3.5-turbo") or (model == "gpt-3.5-turbo-1106"):
        return (0.0015 * usage.prompt_tokens + 0.002 * usage.completion_tokens) / 1000.0
    if model == "gpt-4o":
        return (0.005 * usage.prompt_tokens + 0.015 * usage.completion_tokens) / 1000.0


def call_api(
    client,
    model,
    prompt_messages,
    temperature=1.0,
    max_tokens=100,
    seed=2024,
    json_output=False,
):
    if "claude" in model:
        if json_output:
            prompt = (
                prompt_messages[0]["content"]
                + " Directly output the JSON dict with no additional text. "
            )
            prompt_messages = [{"role": "user", "content": prompt}]
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=prompt_messages,
        )
        cost = calc_price(model, message.usage)
        response = message.content[0].text
    else:
        response_format = {"type": "json_object"} if json_output else {"type": "text"}
        completion = client.chat.completions.create(
            model=model,
            messages=prompt_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            seed=seed,
            response_format=response_format,
        )
        cost = calc_price(model, completion.usage)
        response = completion.choices[0].message.content.strip()

    return response, cost


def call_api_claude(client, model, prompt_messages, temperature=1.0, max_tokens=100):
    message = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=prompt_messages,
    )
    cost = calc_price(model, message.usage)
    response = message.content[0].text

    return response, cost


def cache_output(output, file_name):
    if file_name.endswith(".txt"):
        ## store GPT4 output into a txt file
        with open(file_name, "w") as f:
            f.write(output)
    elif file_name.endswith(".json"):
        ## store GPT4 output into a json file
        with open(file_name, "w") as f:
            json.dump(output, f, indent=4)
    return


def print_idea_json(filename):
    with open(filename, "r") as f:
        idea_json = json.load(f)
    idea = idea_json["final_plan_json"]
    name = idea_json["idea_name"]
    print(name)
    for k, v in idea.items():
        if len(v) > 5:
            print("- " + k)
            print(v.strip() + "\n")


def format_plan_json(experiment_plan_json):
    output_str = ""
    for k, v in experiment_plan_json.items():
        if isinstance(v, str):
            output_str += k + ": " + v.strip() + "\n\n"
        else:
            output_str += k + ": " + "\n"
            for sub_k, sub_v in v.items():
                if isinstance(sub_v, str):
                    output_str += "  - " + sub_k + ": " + sub_v.strip() + "\n"
                else:
                    output_str += "  - " + sub_k + ": " + "\n"
                    for sub_sub_k, sub_sub_v in sub_v.items():
                        output_str += (
                            "    - " + sub_sub_k + ": " + sub_sub_v.strip() + "\n"
                        )
            output_str += "\n"
    return output_str


def shuffle_dict_and_convert_to_string(input_dict):
    # Convert dict items to a list and shuffle
    items = list(input_dict.items())
    random.shuffle(items)

    # Convert back to dict and then to a JSON-formatted string
    shuffled_dict = dict(items)
    json_str = json.dumps(shuffled_dict, indent=4)

    return json_str


def clean_code_output(code_output):
    code_output = code_output.strip()
    if code_output.startswith("```python"):
        code_output = code_output[len("```python") :].strip()
    if code_output.endswith("```"):
        code_output = code_output[: -len("```")].strip()
    return code_output


def concat_reviews(paper_json):
    review_str = ""
    meta_review = paper_json["meta_review"]
    all_reviews = paper_json["reviews"]

    review_str += "Meta Review:\n" + meta_review + "\n\n"
    for idx, review in enumerate(all_reviews):
        review_str += "Reviewer #{}:\n".format(idx + 1) + "\n"
        for key, value in review.items():
            if key in [
                "summary",
                "soundness",
                "contribution",
                "strengths",
                "weaknesse",
                "questions",
                "rating",
                "confidence",
            ]:
                review_str += key + ": " + value["value"] + "\n"
        review_str += "\n"

    return review_str


def avg_score(scores):
    scores = [int(s[0]) for s in scores]
    return sum(scores) / len(scores)


def max_score(scores):
    scores = [int(s[0]) for s in scores]
    return max(scores)


def min_score(scores):
    scores = [int(s[0]) for s in scores]
    return min(scores)


def format_experiment_plan_json(
    experiment_plan_json, indent_level=0, skip_test_cases=True, skip_fallback=True
):
    try:
        # Check if the input is a string, if so, return it directly
        if isinstance(experiment_plan_json, str):
            return experiment_plan_json

        output_str = ""
        indent = "  " * indent_level
        for k, v in experiment_plan_json.items():
            if k == "score":
                continue
            if skip_test_cases and k == "Test Case Examples":
                continue
            if skip_fallback and k == "Fallback Plan":
                continue
            if isinstance(v, (str, int, float)):
                output_str += f"{indent}{k}: {v}\n"
            elif isinstance(v, list):
                output_str += f"{indent}{k}:\n"
                for item in v:
                    if isinstance(item, dict):
                        output_str += format_plan_json(item, indent_level + 1)
                    else:
                        output_str += f"{indent}  - {item}\n"
            elif isinstance(v, dict):
                output_str += f"{indent}{k}:\n"
                output_str += format_plan_json(v, indent_level + 1)
        return output_str
    except Exception as e:
        print("Error in formatting experiment plan json: ", e)
        return ""


## Load the model being evaluated
def load_model(model_name):
    with open("../keys.json", "r") as f:
        keys = json.load(f)

    ANTH_KEY = keys["anthropic_key"]
    OAI_KEY = keys["api_key"]
    ORG_ID = keys["organization_id"]

    if "claude" in model_name:
        client = anthropic.Anthropic(
            api_key=ANTH_KEY,
        )
    else:
        client = OpenAI(organization=ORG_ID, api_key=OAI_KEY)

    return client


## Define the metric
def evaluator(client, model_name, seed, question, gold_label, prediction):
    ## we use the simple evaluator of asking the LLM to judge whether the prediction is correct given the gold label
    prompt = "Given the following question and reference answer, determine if the prediction is correct. Just tell me 'yes' or 'no', nothing else is needed.\n\nQuestion: {}\n\nReference Answer: {}\n\nPrediction: {}\n\n".format(
        question, gold_label, prediction
    )
    prompt_messages = [{"role": "user", "content": prompt}]
    response, _ = call_api(
        client,
        model_name,
        prompt_messages,
        temperature=0.0,
        max_tokens=1,
        seed=seed,
        json_output=False,
    )
    judgment = False
    if response.strip().lower() == "yes":
        return True

    return judgment
