def main():
    # Display the welcome message when the program starts
    print("Rule-Based AI Chatbot")
    print("Type 'help' to see commands.")
    print("Type 'bye' or 'exit' to exit.")
    print("-" * 30)

    # Start a continuous loop
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        # Convert user input to lowercase to make it case-insensitive
        command = user_input.lower().strip()

        # Decision making using if-elif-else logic
        if command == "hi" or command == "hello":
            print("Bot: Hello! Nice to meet you.")
            
        elif command == "how are you":
            print("Bot: I'm functioning perfectly, thank you for asking!")
            
        elif command == "name":
            print("Bot: I am RuleBot, your friendly console chatbot.")
            
        elif command == "age":
            print("Bot: I am as old as the code that created me.")
            
        elif command == "joke":
            print("Bot: Why do programmers prefer dark mode? Because light attracts bugs!")
            
        elif command == "fact":
            print("Bot: Did you know Python was named after the comedy group Monty Python?")
            
        elif command == "weather":
            print("Bot: It's always sunny in the digital world!")
            
        elif command == "time":
            print("Bot: Time is an illusion... but seriously, I don't have a clock.")
            
        elif command == "hobby":
            print("Bot: I enjoy processing text and talking to humans.")
            
        elif command == "help":
            print("Bot: Available commands: hi, hello, how are you, name, age, joke, fact, weather, time, hobby, help, bye, exit")
            
        elif command == "bye" or command == "exit":
            # Print a goodbye message and break the loop to terminate the program
            print("Bot: Goodbye!")
            break
            
        else:
            # Handle unknown commands
            print("Bot: Sorry, I don't understand that command.")

if __name__ == "__main__":
    main()
