from storey import Choice

class MyChoice(Choice):
    def select_outlets(self, event):
        outlets = ["GeneralAgent"]
        if event.results["classification"] == "loans":
            outlets.append("LoanAgent")
        elif event.results["classification"] == "investments":
            outlets.append("InvestmentAgent")
        else:
            outlets.append("GeneralAgent")
        return outlets
