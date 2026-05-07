Output assets/longtrainer-demo.gif
Require python

Set Shell "bash"
Set FontSize 14
Set Width 1200
Set Height 850
Set Padding 30
Set Theme "Catppuccin Mocha"

Hide
Type "source .venv/bin/activate"
Enter
Type "export OPENAI_API_KEY=$OPENAI_API_KEY"
Enter
Type "export VECTORSTORE=qdrant"
Enter
Type "export LLM_MODEL=gpt-5.2"
Enter
Type "export EMBEDDING_MODEL=text-embedding-3-small"
Enter
Type "clear"
Enter
Show

Sleep 1s
Type "python demos/longtrainer_demo.py"
Enter

Sleep 120s
