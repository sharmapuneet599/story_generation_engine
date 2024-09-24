from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
import random

def get_story_types():
   """
   Define and return story types and their probabilities.
   
   Returns:
   tuple: A tuple containing two lists - story types and their probabilities
   """
   types = ["Mystery", "Romance", "Adventure", "Comedy", "Fantasy", "Horror", "Sci-Fi"]
   probabilities = [0.1, 0.1, 0.1, 0.4, 0.1, 0.1, 0.1]
   return types, probabilities

def initialize_model(model_name="gpt-4o", temperature=0.2):
   """
   Initialize and return a ChatOpenAI model.

   Args:
   model_name (str): The name of the model to use
   temperature (float): The temperature setting for the model

   Returns:
   ChatOpenAI: Initialized model
   """
   return ChatOpenAI(
      model=model_name,
      api_key="", # add key here
      temperature=temperature
   )

def get_prompt_template():
   """
   Define and return the prompt template.
   Returns:
   PromptTemplate: The prompt template for story generation
   """
   prompt = """
      You are a creative storyteller tasked with generating a short, engaging conversational story involving the characters provided. Follow these guidelines strictly:

      1. Characters:
         - Use only the character names provided.
         - Characters can walk to other characters, indicated by "A walks to B".
         - Once a character moves, they cannot return to their original location.

      2. Dialogue:
         - Format character speech as "CharacterName: Dialogue text".
         - Keep dialogues concise and relevant to the story.

      3. Actions and Pauses:
         - Include pauses where appropriate, formatted as "Pause for X" where X is the number of seconds.
         - Limit pauses to a maximum of 2 seconds.

      4. Scene Descriptions:
         - Use "Title" to indicate scene changes, beginnings, or endings.
         - Format as: Title "Scene description or story event"
         - Use these for important story beats or to set the scene.

      5. Story Structure:
         - Begin with a Title describing the initial scene or introducing the story.
         - Develop a clear, short plot with a beginning, middle, and end.
         - Conclude with a Title indicating "The End" or a similar closing statement.

      6. General Rules:
         - Keep the story between 10-20 lines, including all elements (dialogue, actions, titles).
         - Ensure each character speaks at least once.
         - Include at least one instance of a character walking to another.
         - Use at least one pause in the story.
         - Maintain a coherent and engaging narrative throughout.

      Output Format:
      Title "Opening scene description or story title"
      [Story content following the above guidelines]
      Title "The End" or a closing statement

      Example Input:
      ["Alice",  "Bob", "Charlie"]

      Example Output:
      Title "A Chance Encounter in the Park"
      Alice walks to Bob
      Alice: Excuse me, have you seen a lost dog around here?
      Bob: No, I haven't. Is it yours?
      Alice: Yes, he ran off while I was reading on the bench.
      Pause for 3
      Charlie walks to Alice
      Charlie: I think I saw a dog running towards the pond!
      Alice: Oh, thank you! I'll check there right away.
      Bob: I'll help you look. Let's go together.
      Title "The Search Begins"

      Your input will be the character names and you have to generate a unique and interesting story using the provided character names, adhering strictly to these guidelines and format.

      Character Names: {character_names}
      Story type: {story_type}
      """
   return PromptTemplate(input_variables=["character_names", "story_type"], template=prompt)

def generate_story(model, characters, story_type, prompt_template):
   """
   Generate a story using the given model, characters, and story type.
   
   Args:
   model (ChatOpenAI): Initialized ChatOpenAI model
   characters (list): List of character names
   story_type (str): Type of story to generate
   prompt_template (PromptTemplate): The prompt template to use
   
   Returns:
   str: Generated story content
   """
   formatted_prompt = prompt_template.format(character_names=characters, story_type=story_type)
   response = model.invoke(formatted_prompt)
   return response.content

def select_random_story_type(types, probabilities):
   """
   Select a random story type based on given probabilities.
   
   Args:
   types (list): List of story types
   probabilities (list): Corresponding probabilities for each story type
   
   Returns:
   str: Selected story type
   """
   return random.choices(types, weights=probabilities)[0]

def main(characters):
   # Initialize components
   model = initialize_model()
   story_types, probabilities = get_story_types()
   prompt_template = get_prompt_template()
   
   # Select story type and characters
   story_type = select_random_story_type(story_types, probabilities)
   print(f"Story type: {story_type}")
   print(f"Characters: {characters}")
   
   # Generate and print the story
   story = generate_story(model, characters, story_type, prompt_template)
   print(story)

if __name__ == "__main__":
   characters = ["Dhoak", "Stuma", "BaronessMadlin"]
   main(characters)
