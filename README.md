
# Corporate Joke

To build an Intelligent corporate joke application using NLP. The model should reply with jokes as
demanded based on circumstances.




## Requirements

1. Create a function that can return jokes without input.
2. Create a function that can accept a joke and respond joke based on the joke provided.
Example:
    Input: How did pirates communicate before the internet?
    Response: Pier to Pier Networking.


## Preparing the Dataset 
### Data Collection
Using PRAW (Python Reddit API Wrapper), we collected over 250,000 Jokes Dataset from the r/Jokes dataset with four attributes. Each being 

### Data Cleaning

The upvotes on the Jokes posted was taken as the deciding factor if the jokes are funny or not. SO, only jokes with more than 1000 upvotes were kept. 
    
    df = df[df['Upvotes'] > 1000]  

Next it was found that in most cases, half the joke was found in the title and the other half in the body, hence both columns were concatenated.

    df['Joke'] = df['Title'].map(str) + ' ' + df['Body'].map(str)
    df.drop(["Title", "Body"], axis = 1, inplace = True)




## Model

We have used HuggingFace Library for GPT-2 Model and the whole code is written in Pytorch.

### Pre-Processing

Open GPT-2 is a transformer type architecture which uses the decoder of transformers .

There are two ways in which data can be presented to the model, depending on the objective you want to achieve

    Joke generator
    Humorous Sentence Completion


#### Joke Generation

In this task the model simply tries to generate jokes, given the length of joke and number of jokes you want it to generate. Here we append 'JOKE:' at the start of every joke in our dataframe and '<|endoftext|>' at the end of each joke which tells our model that our joke has ended. At the time of inference , we simply provide number of jokes and length of each joke and our model will print out jokes based on what it has learned


#### Humorous Sentence Completion

In this our model tries to complete a sentence in a humorous way given any input word or words it has never seen before.

For this task , We took only the Jokes in our dataset which were question,answer types and started with Why,When,How,etc. Then processed the data in this format

<|soq|> question <|sep|> answer <|endoftext|>

It looks like an input to Question answering system , only the whole string is treated as one string , instead of getting different token_type_ids for Questions and Asnwers

### HyperParameters

        BATCH_SIZE = 16
        EPOCHS = 4
        LEARNING_RATE = 3e-5
        MAX_LEN = 64
    

## Web Application
The web application is made on Streamlit.
In order to run the application, in your terminal run:
    
     streamlit run main.py
## Output

![Image description](https://github.com/glitchdawg/Corporate-joker/blob/main/Pictures/Joke%20Generator.png)

![Image description](https://github.com/tanulsingh/Humour.ai/blob/master/Demo/Tanul.PNG)

