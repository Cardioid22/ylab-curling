#!/usr/bin/env python3
"""
Curling Stone Visualization Program
Visualizes stone positions on a curling sheet according to actual game dimensions.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle
import numpy as np

class CurlingSheet:
    # Sheet dimensions (in meters)
    SHEET_WIDTH = 4.75
    SHEET_HEIGHT = 40.234  # Back line

    # House dimensions
    HOUSE_CENTER_X = 0.0
    HOUSE_CENTER_Y = 38.405  # Tee line
    HOUSE_RADIUS = 1.829

    # Lines
    HOG_LINE = 32.004

    # House circles (distances from center)
    CIRCLE_BUTTON = 0.15      # White
    CIRCLE_4FOOT = 0.61       # Red (from 0.15 to 0.61)
    CIRCLE_8FOOT = 1.22       # White (from 0.61 to 1.22)
    CIRCLE_12FOOT = 1.829     # Blue (from 1.22 to 1.829)

    # Stone properties
    STONE_RADIUS = 0.145  # Approximate stone radius

    # Team colors
    TEAM0_COLOR = '#FFD700'  # Gold for Team 0
    TEAM1_COLOR = '#C41E3A'  # Red for Team 1

    def __init__(self, figsize=(8, 10)):
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.stones = []
        self.state_info = {"end": 1, "shot": 0}

    def draw_sheet(self):
        """Draw the curling sheet with house and lines."""
        # Set axis limits (only show from hog line to back line)
        self.ax.set_xlim(-self.SHEET_WIDTH/2, self.SHEET_WIDTH/2)
        self.ax.set_ylim(self.HOG_LINE - 0.5, self.SHEET_HEIGHT + 0.5)
        self.ax.set_aspect('equal')

        # Draw sheet boundary
        sheet_rect = patches.Rectangle(
            (-self.SHEET_WIDTH/2, 0),
            self.SHEET_WIDTH,
            self.SHEET_HEIGHT,
            linewidth=2,
            edgecolor='black',
            facecolor='lightblue',
            alpha=0.3
        )
        self.ax.add_patch(sheet_rect)

        # Draw house circles (from outside to inside)
        # 12-foot circle (Blue)
        circle_12 = Circle(
            (self.HOUSE_CENTER_X, self.HOUSE_CENTER_Y),
            self.CIRCLE_12FOOT,
            color='blue',
            alpha=0.4,
            zorder=1
        )
        self.ax.add_patch(circle_12)

        # 8-foot circle (White)
        circle_8 = Circle(
            (self.HOUSE_CENTER_X, self.HOUSE_CENTER_Y),
            self.CIRCLE_8FOOT,
            color='white',
            alpha=0.6,
            zorder=2
        )
        self.ax.add_patch(circle_8)

        # 4-foot circle (Red)
        circle_4 = Circle(
            (self.HOUSE_CENTER_X, self.HOUSE_CENTER_Y),
            self.CIRCLE_4FOOT,
            color='red',
            alpha=0.4,
            zorder=3
        )
        self.ax.add_patch(circle_4)

        # Button (White)
        button = Circle(
            (self.HOUSE_CENTER_X, self.HOUSE_CENTER_Y),
            self.CIRCLE_BUTTON,
            color='white',
            alpha=0.8,
            zorder=4
        )
        self.ax.add_patch(button)

        # Draw tee line
        self.ax.axhline(
            y=self.HOUSE_CENTER_Y,
            color='black',
            linestyle='--',
            linewidth=1.5,
            label='Tee Line'
        )

        # Draw hog line
        self.ax.axhline(
            y=self.HOG_LINE,
            color='gray',
            linestyle='--',
            linewidth=1,
            label='Hog Line'
        )

        # Draw back line
        self.ax.axhline(
            y=self.SHEET_HEIGHT,
            color='black',
            linestyle='-',
            linewidth=2,
            label='Back Line'
        )

        # Draw center line
        self.ax.axvline(
            x=0,
            color='gray',
            linestyle=':',
            linewidth=1,
            alpha=0.5
        )

        # Labels
        self.ax.set_xlabel('X (meters)', fontsize=10)
        self.ax.set_ylabel('Y (meters)', fontsize=10)
        self.ax.grid(True, alpha=0.2)

    def add_stone(self, x, y, team):
        """
        Add a stone to the sheet.

        Args:
            x (float): X coordinate in meters
            y (float): Y coordinate in meters
            team (int): Team number (0 or 1)
        """
        self.stones.append({'x': x, 'y': y, 'team': team})

    def set_state(self, end, shot):
        """
        Set the current game state.

        Args:
            end (int): End number
            shot (int): Shot number
        """
        self.state_info['end'] = end
        self.state_info['shot'] = shot

    def draw_stones(self):
        """Draw all stones on the sheet."""
        for stone in self.stones:
            color = self.TEAM0_COLOR if stone['team'] == 0 else self.TEAM1_COLOR

            # Draw stone
            stone_circle = Circle(
                (stone['x'], stone['y']),
                self.STONE_RADIUS,
                color=color,
                edgecolor='black',
                linewidth=2,
                zorder=10,
                alpha=0.9
            )
            self.ax.add_patch(stone_circle)

            # Add team label on stone
            self.ax.text(
                stone['x'], stone['y'],
                f"T{stone['team']}",
                ha='center', va='center',
                fontsize=8,
                fontweight='bold',
                color='white',
                zorder=11
            )

    def draw_title(self):
        """Draw title with state information."""
        title = f"Curling Stone Positions - End {self.state_info['end']}, Shot {self.state_info['shot']}"
        self.ax.set_title(title, fontsize=12, fontweight='bold', pad=10)

        # Add legend for teams
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=self.TEAM0_COLOR, edgecolor='black', label='Team 0'),
            Patch(facecolor=self.TEAM1_COLOR, edgecolor='black', label='Team 1')
        ]
        self.ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    def show(self):
        """Display the plot."""
        self.draw_sheet()
        self.draw_stones()
        self.draw_title()
        plt.tight_layout()
        plt.show()

    def save(self, filename):
        """Save the plot to a file."""
        self.draw_sheet()
        self.draw_stones()
        self.draw_title()
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved to: {filename}")


def interactive_mode():
    """Interactive mode to add stones manually."""
    print("=" * 60)
    print("Curling Stone Visualization - Interactive Mode")
    print("=" * 60)

    # Create sheet
    sheet = CurlingSheet()

    # Get state info
    end = int(input("Enter end number (default=1): ") or "1")
    shot = int(input("Enter shot number (default=0): ") or "0")
    sheet.set_state(end, shot)

    print("\n" + "=" * 60)
    print("Add stones to the sheet")
    print("Enter 'done' to finish and display")
    print("=" * 60)

    while True:
        print("\nAdd stone:")
        x_input = input("  X coordinate (meters, or 'done' to finish): ")

        if x_input.lower() == 'done':
            break

        try:
            x = float(x_input)
            y = float(input("  Y coordinate (meters): "))
            team = int(input("  Team (0 or 1): "))

            if team not in [0, 1]:
                print("  Error: Team must be 0 or 1")
                continue

            sheet.add_stone(x, y, team)
            print(f"  Added stone for Team {team} at ({x}, {y})")

        except ValueError:
            print("  Error: Invalid input. Please enter numbers.")

    # Display
    print("\nDisplaying curling sheet...")
    sheet.show()


def example_states():
    """Show example stone configurations from simple_experiment.cpp."""
    import os

    # Create output directory if it doesn't exist
    output_dir = "./TestPosition"
    os.makedirs(output_dir, exist_ok=True)

    # State 1: Initial (shot 0) - No stones
    sheet1 = CurlingSheet()
    sheet1.set_state(1, 0)
    sheet1.save(f"{output_dir}/state_shot0.png")

    # State 2: Guard stone (shot 2)
    sheet2 = CurlingSheet()
    sheet2.set_state(1, 2)
    sheet2.add_stone(0.3, 37.0, 0)
    sheet2.save(f"{output_dir}/state_shot2.png")

    # State 3: Mid-game (shot 4)
    sheet3 = CurlingSheet()
    sheet3.set_state(1, 4)
    sheet3.add_stone(0.3, 37.0, 0)
    sheet3.add_stone(-0.4, 37.2, 1)
    sheet3.add_stone(0.2, 38.5, 0)
    sheet3.save(f"{output_dir}/state_shot4.png")

    # State 4: Complex mid-game (shot 6)
    sheet4 = CurlingSheet()
    sheet4.set_state(1, 6)
    sheet4.add_stone(0.3, 37.0, 0)
    sheet4.add_stone(-0.4, 37.2, 1)
    sheet4.add_stone(0.2, 38.5, 0)
    sheet4.add_stone(-0.3, 38.3, 1)
    sheet4.add_stone(0.8, 37.5, 0)
    sheet4.save(f"{output_dir}/state_shot6.png")

    # State 5: End-game (shot 8)
    sheet5 = CurlingSheet()
    sheet5.set_state(1, 8)
    sheet5.add_stone(0.3, 37.0, 0)
    sheet5.add_stone(-0.4, 37.2, 1)
    sheet5.add_stone(0.2, 38.5, 0)
    sheet5.add_stone(-0.3, 38.3, 1)
    sheet5.add_stone(0.8, 37.5, 0)
    sheet5.add_stone(0.0, 38.405, 1)
    sheet5.add_stone(1.0, 37.8, 0)
    sheet5.save(f"{output_dir}/state_shot8.png")

    print(f"Example states saved to {output_dir}!")


def main():
    """Main function."""
    print("Curling Stone Visualization Program")
    print("1. Interactive mode (add stones manually)")
    print("2. Generate example states from simple_experiment.cpp")

    choice = input("\nSelect mode (1 or 2): ")

    if choice == "1":
        interactive_mode()
    elif choice == "2":
        example_states()
    else:
        print("Invalid choice. Exiting.")


if __name__ == '__main__':
    main()
