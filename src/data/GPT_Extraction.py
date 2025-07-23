from openai import OpenAI
import argparse
import os 
import logging
import glob
import numpy as np

# Define model to use
MODEL = "o4-mini-2025-04-16" 

########################################################################################################
def CallGPT(text, prompt, client, MODEL):
    messages=[
        {"role": "system", "content": prompt},
        {"role": "user", "content": text },
    ]

    # Proceed with API call only if token count >= 10,000
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            messages=messages,  
            store=True,)
    except Exception as e:
        print(f"Error during API call: {e}")
        logging.error(f"Error during API call: {e}")
        return f"Error: {e}"
    
    print(f"API call successful. Model: {MODEL}, Tokens used: {completion.usage.total_tokens}")
    logging.info(f"API call successful. Model: {MODEL}, Tokens used: {completion.usage.total_tokens}")

    return completion.choices[0].message.content
########################################################################################################

########################################################################################################
    
########################################################################################################


########################################################################################################
def main():

    parser = argparse.ArgumentParser(description="Extract metadata in SDRF format from a manuscript using OpenAI API")
    parser.add_argument("--inpath", required=True, type=str, help="Path to a directory containing text publications ending in .txt")
    parser.add_argument("--prompt", required=True, type=str, help="Path to the prompt file that will be used to extract metadata from the manuscript")
    parser.add_argument("--outpath", required=True, type=str, help="Optional single file to process")
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    OUTPUT_DIR = os.path.join(args.outpath, MODEL)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f'MADE: {OUTPUT_DIR}')

    ## initiate the OpenAI client with my super secrete hard coded API key o.O
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    
    ## load the prompt file if provided
    if args.prompt:
        if not os.path.isfile(args.prompt):
            ValueError(f'Prompt file {args.prompt} does not exist. Exiting.')
        with open(args.prompt, 'r') as f:
            prompt = f.read()
        print(f'Loaded prompt from {args.prompt}')
        logging.info(f'Loaded prompt from {args.prompt}')
    else:
        raise ValueError('No prompt file provided. Exiting...')
    print(f'{"#"*50}\nPROMPT:\n{prompt}\n{"#"*50}')
    #############################################################

    #############################################################  
    if os.path.isfile(args.inpath): ## is a file path and should be to a SQL database and should end in .db
        print(f'Loading the input .pkl file: {args.inpath}')
        with open(args.inpath, 'r') as f:
            text = f.read()
        print(f'Extracting metadata from {args.inpath} using prompt {args.prompt}')

        metadata = CallGPT(text, prompt, client, MODEL)
        print(f'Extracted metadata: {metadata}')
        output_file = os.path.join(OUTPUT_DIR, os.path.basename(args.inpath) + '_GPTextract.txt')
        with open(output_file, 'w') as out_f:
            out_f.write(metadata)
        print(f'Metadata saved to {output_file}')
    
    elif os.path.isdir(args.inpath): ## is a directory and should contain .txt files
        print(f'Loading the input directory: {args.inpath}')
        files = glob.glob(os.path.join(args.inpath, '*.txt'))
        if not files:
            raise ValueError(f'No .txt files found in {args.inpath}. Exiting...')
        
        for file in files:
            print(f'Processing file: {file}')
            with open(file, 'r') as f:
                text = f.read()
            print(f'Extracting metadata from {file} using prompt {args.prompt}')

            metadata = CallGPT(text, prompt, client, MODEL)
            print(f'Extracted metadata: {metadata}')
            output_file = os.path.join(OUTPUT_DIR, os.path.basename(file) + '_GPTextract.txt')
            with open(output_file, 'w') as out_f:
                out_f.write(metadata)
            print(f'Metadata saved to {output_file}')
    #############################################################

if __name__ == "__main__":
    main()
print('NORMAL TERMINATION')
