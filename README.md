# LongTrainer - Produciton Ready LangChain

## Features

1. Long Memory <br>
2. Unique Bots/Chat Management <br>
2. Enhanced Customization </br>
3. Memory Management <br>
4. GPT Vision Support <br>
5. Different Data Formats <br>
6. VectoreStore Management <br>

## Usage Example

        from longtrainer import LongTrainer
        os.environ["OPENAI_API_KEY"] = "sk-"
        trainer = LongTrainer()
        bot_id = trainer.initialize_bot_id()
        print('Bot ID: ', bot_id)

        # Add Data
        trainer.add_document_from_path(path, bot_id)

        # Initialize Bot
        trainer.create_bot(bot_id)

        # Initialize new Chat
        chat_id = trainer.new_chat(bot_id)

        # Query
        response = trainer._get_response(query, chat_id, bot_id)



