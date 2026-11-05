from dataclasses import dataclass


@dataclass
class Bankroll:
    cash: int
    committed: int = 0

    def commit(self, amount: int):
        if amount <= 0 or amount > self.cash:
            raise ValueError("Wager must be positive and funded")
        self.cash -= amount
        self.committed += amount

    def settle(self, returned: int):
        if returned < 0 or not self.committed:
            raise ValueError("Invalid settlement")
        self.cash += returned
        self.committed = 0
