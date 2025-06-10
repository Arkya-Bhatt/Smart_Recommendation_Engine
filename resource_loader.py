import pandas as pd
import ast

def parse_tags(tags_str):
    if isinstance(tags_str, str):
        try:
            tags_list = ast.literal_eval(tags_str)
            if isinstance(tags_list, list):
                return [str(tag).strip().lower() for tag in tags_list]
        except (ValueError, SyntaxError):
            pass
    return []


def load_resources_from_csv(csv_filepath):
    try:
        df = pd.read_csv(csv_filepath)
    except FileNotFoundError:
        print(f"Error: The file {csv_filepath} was not found.")
        return []
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return []

    processed_resources = []
    for index, row in df.iterrows():
        tags_raw = row.get('_topics', "")
        custom_topics_raw = row.get('customtopics', "")
        
        parsed_tags = parse_tags(tags_raw)
        parsed_custom_topics = parse_tags(custom_topics_raw)
        
        all_tags = list(set(parsed_tags + parsed_custom_topics))

        description_raw = row.get('_github_description', row.get('description', ''))
        description_str = str(description_raw) if pd.notna(description_raw) else ""

        resource = {
            "id": row.get('', index),
            "name": row.get('_reponame', 'N/A'),
            "type": str(row.get('category', 'general')).lower(),
            "url": row.get('githuburl', ''),
            "description": description_str,
            "tags": all_tags,
            "stars": row.get('_stars', 0),
            "stars_per_week": row.get('_stars_per_week', 0.0),
            "pop_score": row.get('_pop_score', 0.0),
            "language": row.get('_language', '')
        }
        processed_resources.append(resource)
    
    return processed_resources