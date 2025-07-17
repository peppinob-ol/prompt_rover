"""
Root launcher for Prompt Rover (used by Hugging Face Spaces and for local runs).

Simply forwards execution to the real package entry-point
`prompt_rover.app.main()`.
"""

from prompt_rover.app import main

if __name__ == "__main__":
    main() 