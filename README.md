# Movie-Recommendation-System

# MovieMagic Recommender

A stylish and powerful movie recommendation system with a modern GUI interface that suggests movies based on content similarity, user preferences, or a hybrid approach.

##  Features

- **Multiple Recommendation Methods**:
  - Content-based filtering (find similar movies)
  - Feature-based filtering (filter by genre, studio, scores)
  - Hybrid approach (combines both methods)

- **Beautiful Dark Theme UI** with:
  - Modern color scheme
  - Emoji-enhanced interface
  - Responsive layout
  - Interactive controls

- **Comprehensive Movie Data** including:
  - Title, Genre, Studio
  - Audience and critic scores
  - Financial performance metrics

- **Visual Analytics**:
  - Genre distribution charts
  - Score comparisons
  - Interactive results tables

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/movie-recommender.git
   cd movie-recommender
   ```

2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Place your `movies.csv` file in the project directory

## 🚀 Usage

Run the application:
```bash
python movie_recommender.py
```

### How to Use:
1. Select a recommendation type
2. Choose a movie or set filters
3. Click "Get Recommendations"
4. View results in the popup window
5. Explore genre visualizations

## File Structure

```
movie-recommender/
├── movie_recommender.py    # Main application code
├── movies.csv              # Movie dataset (sample provided)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── screenshot.png          # Application screenshot
```

## Requirements

- Python 3.7+
- Required packages:
  - pandas
  - scikit-learn
  - tkinter
  - matplotlib
  - seaborn
  - pillow

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your improvements.
