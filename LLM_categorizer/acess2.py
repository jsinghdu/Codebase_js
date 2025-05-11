import pandas as pd

class CourseValidator:
    def __init__(self, input_csv, output_csv):
        """
        Initialize the CourseValidator with input and output CSV file paths.
        :param input_csv: Path to the input CSV file.
        :param output_csv: Path to the output CSV file.
        """
        self.input_csv = input_csv
        self.output_csv = output_csv
        self.df = None

    def load_data(self):
        """
        Load the CSV data into a DataFrame.
        """
        try:
            self.df = pd.read_csv(self.input_csv)
            print(f"Data successfully loaded from {self.input_csv}")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def flag_low_scores(self, bert_threshold=50, bertscore_threshold=35):
        """
        Add a flag column for manual review based on low BERT and BERTScore thresholds.
        :param bert_threshold: Threshold for BERT Similarity percentage.
        :param bertscore_threshold: Threshold for BERTScore percentage.
        """
        if self.df is not None:
            self.df["Needs_Review"] = (self.df["BERT_Similarity (%)"] < bert_threshold) | \
                                      (self.df["BERTScore (%)"] < bertscore_threshold)
        else:
            raise ValueError("Dataframe is not loaded. Call load_data() first.")

    def sort_by_similarity(self):
        """
        Sort the DataFrame by BERT Similarity percentage.
        """
        if self.df is not None:
            self.df = self.df.sort_values(by="BERT_Similarity (%)", ascending=False)
        else:
            raise ValueError("Dataframe is not loaded. Call load_data() first.")

    def display_matches(self, top_n=10):
        """
        Print the top matches and the bottom matches that need review.
        :param top_n: Number of top matches to display.
        """
        if self.df is not None:
            print("Top Matches:")
            print(self.df.head(top_n))

            print("\nBottom Matches (needs review):")
            print(self.df[self.df["Needs_Review"]].head(top_n))
        else:
            raise ValueError("Dataframe is not loaded. Call load_data() first.")

    def save_results(self):
        """
        Save the flagged data to the output CSV file.
        """
        if self.df is not None:
            self.df.to_csv(self.output_csv, index=False)
            print(f"Results saved to {self.output_csv}")
        else:
            raise ValueError("Dataframe is not loaded. Call load_data() first.")


# Usage
if __name__ == "__main__":
    validator = CourseValidator(input_csv="validated_courses.csv", output_csv="validated_courses_with_flags.csv")
    
    validator.load_data()
    validator.flag_low_scores(bert_threshold=50, bertscore_threshold=35)
    validator.sort_by_similarity()
    validator.display_matches(top_n=10)
    validator.save_results()
