# import ollama
#
# stream = ollama.chat(
#     model='llama3.1:8b',
#     messages=[{'role': 'user', 'content': 'Why is the sky blue?'}],
#     stream=True,
# )
#
# for chunk in stream:
#   print(chunk['message']['content'], end='', flush=True)

import asyncio
import ollama



async def main():

    client = ollama.AsyncClient()

    messages = []

    while True:
        if content_in := input('>>> '):
            messages.append({'role': 'user', 'content': content_in})

            content_out = ''
            message = {'role': 'assistant', 'content': ''}
            async for response in await client.chat(model='llama3.1:8b', messages=messages, stream=True):
                if response['done']:
                    messages.append(message)

                content = response['message']['content']
                print(content, end='', flush=True)

                content_out += content

                message['content'] += content


            print()
        else:
            break


try:
    asyncio.run(main())
except (KeyboardInterrupt, EOFError):
    ...
