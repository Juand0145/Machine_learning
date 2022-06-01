#!/usr/bin/env python3
"""FIle that create and infinite loop"""

bye = ["exit", "quit", "goodbye", "bye"]
while True:
  ask = input("Q: ").lower()

  if ask in bye:
    print(f"A: Goodbye")
    break
  else: 
    print(f"A: ")
