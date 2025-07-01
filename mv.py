import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import io
import matplotlib.pyplot as plt
import seaborn as sns
import os

class StyledMovieRecommender:
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 MovieMagic Recommender")
        self.root.geometry("900x700")
        self.root.configure(bg="#2c3e50")
        
        # Set theme colors
        self.bg_color = "#2c3e50"
        self.primary_color = "#3498db"
        self.secondary_color = "#2980b9"
        self.accent_color = "#e74c3c"
        self.text_color = "#ecf0f1"
        self.entry_bg = "#34495e"
        
        # Set style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color, font=('Helvetica', 10))
        self.style.configure('TButton', background=self.primary_color, foreground=self.text_color, 
                           font=('Helvetica', 10, 'bold'), borderwidth=1)
        self.style.map('TButton', background=[('active', self.secondary_color)])
        self.style.configure('TRadiobutton', background=self.bg_color, foreground=self.text_color)
        self.style.configure('TCombobox', fieldbackground=self.entry_bg, foreground=self.text_color)
        self.style.configure('TScale', background=self.bg_color)
        self.style.configure('TLabelFrame', background=self.bg_color, foreground=self.text_color)
        self.style.configure('Treeview', background=self.entry_bg, fieldbackground=self.entry_bg, 
                           foreground=self.text_color, rowheight=25)
        self.style.configure('Treeview.Heading', background=self.primary_color, foreground=self.text_color)
        self.style.map('Treeview', background=[('selected', self.secondary_color)])
        
        # Set the CSV file path
        self.csv_path = "D:/Project/Movies Recommendation system/movies.csv"
        
        # Check if file exists
        if not os.path.exists(self.csv_path):
            messagebox.showerror("Error", f"CSV file not found at:\n{self.csv_path}")
            self.root.destroy()
            return
        
        # Load data
        try:
            self.movies = self._load_and_preprocess_data(self.csv_path)
            self._build_models()
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data:\n{str(e)}")
            self.root.destroy()
            return
        
        # Create GUI
        self._create_widgets()
    
    def _load_and_preprocess_data(self, file_path):
        """Load and preprocess the movie data"""
        movies = pd.read_csv(file_path)
        
        # Clean data
        movies['Worldwide Gross'] = movies['Worldwide Gross'].str.replace(r'[\$, ]', '', regex=True).astype(float)
        movies['Profitability'] = movies['Profitability'].replace(0, np.nan)
        
        # Clean Year column
        if pd.api.types.is_datetime64_any_dtype(movies['Year']):
            movies['Year'] = movies['Year'].dt.year
        elif isinstance(movies['Year'].iloc[0], str):
            try:
                movies['Year'] = movies['Year'].str.extract(r'(\d{4})').astype(float)
            except:
                pass
        
        # Create combined features
        movies['features'] = (
            movies['Film'] + ' ' + 
            movies['Genre'] + ' ' + 
            movies['Lead Studio'].fillna('') + ' ' + 
            movies['Year'].astype(str)
        )
        
        return movies
    
    def _build_models(self):
        """Build recommendation models"""
        self.tfidf = TfidfVectorizer(stop_words='english')
        self.tfidf_matrix = self.tfidf.fit_transform(self.movies['features'])
        self.content_similarity = cosine_similarity(self.tfidf_matrix, self.tfidf_matrix)
        
        numerical_features = ['Audience score %', 'Rotten Tomatoes %', 'Worldwide Gross', 'Profitability']
        self.scaler = MinMaxScaler()
        self.movies[numerical_features] = self.scaler.fit_transform(
            self.movies[numerical_features].fillna(self.movies[numerical_features].median()))
    
    def _create_widgets(self):
        """Create styled GUI widgets"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding=(20, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        title_label = ttk.Label(header_frame, 
                              text="🎬 MovieMagic Recommender", 
                              font=('Helvetica', 18, 'bold'),
                              foreground=self.primary_color)
        title_label.pack(side=tk.LEFT)
        
        # Content frame
        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - controls
        left_panel = ttk.Frame(content_frame, width=300, padding=(0, 0, 20, 0))
        left_panel.pack(side=tk.LEFT, fill=tk.Y)
        
        # Recommendation type
        type_frame = ttk.LabelFrame(left_panel, text=" Recommendation Type ", padding=10)
        type_frame.pack(fill=tk.X, pady=(0, 15))
        
        self.rec_type = tk.StringVar(value="content")
        ttk.Radiobutton(type_frame, text="🎥 Content-Based", variable=self.rec_type, 
                        value="content").pack(anchor=tk.W, pady=3)
        ttk.Radiobutton(type_frame, text="🔍 Feature-Based", variable=self.rec_type, 
                        value="feature").pack(anchor=tk.W, pady=3)
        ttk.Radiobutton(type_frame, text="✨ Hybrid", variable=self.rec_type, 
                        value="hybrid").pack(anchor=tk.W, pady=3)
        
        # Movie selection
        movie_frame = ttk.LabelFrame(left_panel, text=" Movie Selection ", padding=10)
        movie_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(movie_frame, text="Select a Movie:").pack(anchor=tk.W)
        self.movie_var = tk.StringVar()
        self.movie_combobox = ttk.Combobox(movie_frame, textvariable=self.movie_var, 
                                          values=self.movies['Film'].tolist())
        self.movie_combobox.pack(fill=tk.X, pady=5)
        
        # Filters
        filters_frame = ttk.LabelFrame(left_panel, text=" Recommendation Filters ", padding=10)
        filters_frame.pack(fill=tk.X, pady=(0, 15))
        
        # Genre filter
        ttk.Label(filters_frame, text="Genre:").pack(anchor=tk.W)
        self.genre_var = tk.StringVar()
        self.genre_combobox = ttk.Combobox(filters_frame, textvariable=self.genre_var, 
                                          values=sorted(self.movies['Genre'].unique().tolist()))
        self.genre_combobox.pack(fill=tk.X, pady=5)
        
        # Studio filter
        ttk.Label(filters_frame, text="Studio:").pack(anchor=tk.W)
        self.studio_var = tk.StringVar()
        self.studio_combobox = ttk.Combobox(filters_frame, textvariable=self.studio_var, 
                                           values=sorted(self.movies['Lead Studio'].unique().tolist()))
        self.studio_combobox.pack(fill=tk.X, pady=5)
        
        # Score filters
        score_frame = ttk.Frame(filters_frame)
        score_frame.pack(fill=tk.X, pady=5)
        
        # Audience score
        ttk.Label(score_frame, text="Audience Score ≥").pack(side=tk.LEFT)
        self.audience_score_var = tk.DoubleVar(value=50)
        self.audience_score_label = ttk.Label(score_frame, text="50%", width=5)
        self.audience_score_label.pack(side=tk.RIGHT)
        self.audience_score_slider = ttk.Scale(filters_frame, from_=0, to=100, 
                                              variable=self.audience_score_var,
                                              command=lambda e: self._update_slider_labels())
        self.audience_score_slider.pack(fill=tk.X, pady=5)
        
        # Critic score
        ttk.Label(score_frame, text="Critic Score ≥").pack(side=tk.LEFT)
        self.critic_score_var = tk.DoubleVar(value=50)
        self.critic_score_label = ttk.Label(score_frame, text="50%", width=5)
        self.critic_score_label.pack(side=tk.RIGHT)
        self.critic_score_slider = ttk.Scale(filters_frame, from_=0, to=100, 
                                           variable=self.critic_score_var,
                                           command=lambda e: self._update_slider_labels())
        self.critic_score_slider.pack(fill=tk.X, pady=5)
        
        # Number of recommendations
        num_rec_frame = ttk.Frame(left_panel)
        num_rec_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(num_rec_frame, text="Number of Recommendations:").pack(side=tk.LEFT)
        self.num_rec_var = tk.IntVar(value=5)
        self.num_rec_spinbox = ttk.Spinbox(num_rec_frame, from_=1, to=20, 
                                         textvariable=self.num_rec_var, width=5)
        self.num_rec_spinbox.pack(side=tk.RIGHT)
        
        # Recommendation button
        self.recommend_btn = ttk.Button(left_panel, text="🎯 Get Recommendations", 
                                       command=self._show_recommendations,
                                       style='Accent.TButton')
        self.recommend_btn.pack(fill=tk.X, pady=20)
        
        # Right panel - placeholder for results or info
        right_panel = ttk.Frame(content_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Info text
        info_text = tk.Text(right_panel, wrap=tk.WORD, bg=self.entry_bg, fg=self.text_color,
                          font=('Helvetica', 10), padx=10, pady=10, borderwidth=0)
        info_text.insert(tk.END, "🌟 Welcome to MovieMagic Recommender!\n\n")
        info_text.insert(tk.END, "1. Select a recommendation type\n")
        info_text.insert(tk.END, "2. Choose a movie or set filters\n")
        info_text.insert(tk.END, "3. Click 'Get Recommendations'\n\n")
        info_text.insert(tk.END, "✨ Discover your next favorite movie!")
        info_text.configure(state='disabled')
        info_text.pack(fill=tk.BOTH, expand=True)
        
        # Create custom style for accent button
        self.style.configure('Accent.TButton', background=self.accent_color)
        self.style.map('Accent.TButton', background=[('active', '#c0392b')])
    
    def _update_slider_labels(self, *args):
        """Update slider value labels"""
        self.audience_score_label.config(text=f"{self.audience_score_var.get():.0f}%")
        self.critic_score_label.config(text=f"{self.critic_score_var.get():.0f}%")
    
    def _show_recommendations(self):
        """Show recommendations in a new window"""
        rec_type = self.rec_type.get()
        num_rec = self.num_rec_var.get()
        
        try:
            if rec_type == "content":
                movie_title = self.movie_var.get()
                if not movie_title:
                    messagebox.showerror("Error", "Please select a movie for content-based recommendations")
                    return
                recommendations = self._get_content_based_recommendations(movie_title, num_rec)
                title = f"🎬 Movies similar to {movie_title}"
                
            elif rec_type == "feature":
                preferences = {
                    'Genre': self.genre_var.get() if self.genre_var.get() else None,
                    'Studio': self.studio_var.get() if self.studio_var.get() else None,
                    'Min Audience Score': self.audience_score_var.get(),
                    'Min Critic Score': self.critic_score_var.get()
                }
                recommendations = self._get_feature_based_recommendations(preferences, num_rec)
                title = "🔍 Feature-Based Recommendations"
                
            elif rec_type == "hybrid":
                movie_title = self.movie_var.get()
                if not movie_title:
                    messagebox.showerror("Error", "Please select a movie for hybrid recommendations")
                    return
                preferences = {
                    'Genre': self.genre_var.get() if self.genre_var.get() else None,
                    'Studio': self.studio_var.get() if self.studio_var.get() else None,
                    'Min Audience Score': self.audience_score_var.get(),
                    'Min Critic Score': self.critic_score_var.get()
                }
                recommendations = self._get_hybrid_recommendations(movie_title, preferences, num_rec)
                title = f"✨ Hybrid Recommendations based on {movie_title}"
            
            if recommendations.empty:
                messagebox.showinfo("No Results", "No recommendations found with the current filters")
                return
            
            # Create results window
            self._create_results_window(title, recommendations)
            
        except Exception as e:
            messagebox.showerror("Error", f"An error occurred: {str(e)}")
    
    def _get_content_based_recommendations(self, movie_title, top_n=5):
        """Get content-based recommendations"""
        try:
            idx = self.movies[self.movies['Film'] == movie_title].index[0]
            sim_scores = list(enumerate(self.content_similarity[idx]))
            sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
            sim_scores = sim_scores[1:top_n+1]
            
            movie_indices = [i[0] for i in sim_scores]
            return self.movies.iloc[movie_indices][['Film', 'Genre', 'Lead Studio', 'Year', 'Audience score %', 'Rotten Tomatoes %']]
        except:
            return pd.DataFrame()
    
    def _get_feature_based_recommendations(self, preferences, top_n=5):
        """Get feature-based recommendations"""
        filtered = self.movies.copy()
        
        if preferences.get('Genre'):
            filtered = filtered[filtered['Genre'] == preferences['Genre']]
        if preferences.get('Studio'):
            filtered = filtered[filtered['Lead Studio'] == preferences['Studio']]
        if preferences.get('Min Audience Score'):
            filtered = filtered[filtered['Audience score %'] >= preferences['Min Audience Score']/100]
        if preferences.get('Min Critic Score'):
            filtered = filtered[filtered['Rotten Tomatoes %'] >= preferences['Min Critic Score']/100]
        
        if not filtered.empty:
            filtered['Recommendation Score'] = (
                filtered['Audience score %'] * 0.6 + 
                filtered['Rotten Tomatoes %'] * 0.4
            )
            return filtered.sort_values('Recommendation Score', ascending=False).head(top_n)[
                ['Film', 'Genre', 'Lead Studio', 'Year', 'Audience score %', 'Rotten Tomatoes %']
            ]
        return pd.DataFrame()
    
    def _get_hybrid_recommendations(self, movie_title, preferences, top_n=5):
        """Get hybrid recommendations"""
        content_recs = self._get_content_based_recommendations(movie_title, top_n)
        feature_recs = self._get_feature_based_recommendations(preferences, top_n)
        
        combined = pd.concat([content_recs, feature_recs]).drop_duplicates(subset=['Film'])
        
        if not content_recs.empty and not feature_recs.empty:
            combined['Priority'] = 0
            combined.loc[combined['Film'].isin(content_recs['Film']), 'Priority'] = 1
            combined = combined.sort_values(['Priority', 'Audience score %'], ascending=[False, False])
        
        return combined.head(top_n)
    
    def _create_results_window(self, title, recommendations):
        """Create a styled results window"""
        results_window = tk.Toplevel(self.root)
        results_window.title(title)
        results_window.geometry("1000x700")
        results_window.configure(bg=self.bg_color)
        
        # Main frame
        main_frame = ttk.Frame(results_window, padding=(15, 10))
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header
        header_frame = ttk.Frame(main_frame)
        header_frame.pack(fill=tk.X, pady=(0, 15))
        
        title_label = ttk.Label(header_frame, text=title, 
                              font=('Helvetica', 14, 'bold'),
                              foreground=self.primary_color)
        title_label.pack(side=tk.LEFT)
        
        # Results frame
        results_frame = ttk.Frame(main_frame)
        results_frame.pack(fill=tk.BOTH, expand=True)
        
        # Treeview with scrollbars
        tree_frame = ttk.Frame(results_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        columns = list(recommendations.columns)
        tree = ttk.Treeview(tree_frame, columns=columns, show='headings', height=15)
        
        # Configure columns
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, width=120, anchor=tk.CENTER)
        
        # Add data
        for _, row in recommendations.iterrows():
            tree.insert('', tk.END, values=list(row))
        
        # Add scrollbars
        v_scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=tree.yview)
        h_scroll = ttk.Scrollbar(tree_frame, orient=tk.HORIZONTAL, command=tree.xview)
        tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)
        
        # Grid layout
        tree.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        tree_frame.columnconfigure(0, weight=1)
        tree_frame.rowconfigure(0, weight=1)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        if 'Genre' in recommendations.columns:
            vis_btn = ttk.Button(button_frame, text="📊 Show Genre Distribution", 
                                command=lambda: self._show_genre_visualization(recommendations),
                                style='Accent.TButton')
            vis_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ttk.Button(button_frame, text="❌ Close", 
                              command=results_window.destroy)
        close_btn.pack(side=tk.RIGHT, padx=5)
    
    def _show_genre_visualization(self, recommendations):
        """Show styled genre distribution visualization"""
        # Create figure with dark theme
        plt.style.use('dark_background')
        fig, ax = plt.subplots(figsize=(8, 4), facecolor='#2c3e50')
        ax.set_facecolor('#2c3e50')
        
        # Custom color palette
        palette = sns.color_palette("husl", len(recommendations['Genre'].unique()))
        
        # Create plot
        sns.countplot(data=recommendations, y='Genre', 
                     order=recommendations['Genre'].value_counts().index,
                     palette=palette, ax=ax)
        
        plt.title('Genre Distribution in Recommendations', color='white', pad=20)
        plt.tight_layout()
        
        # Save to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor=fig.get_facecolor())
        buf.seek(0)
        plt.close()
        
        # Create visualization window
        vis_window = tk.Toplevel(self.root)
        vis_window.title("Genre Distribution")
        vis_window.geometry("800x500")
        vis_window.configure(bg=self.bg_color)
        
        # Display image
        img = Image.open(buf)
        photo = ImageTk.PhotoImage(img)
        
        label = ttk.Label(vis_window, image=photo)
        label.image = photo
        label.pack(pady=10)
        
        # Close button
        close_btn = ttk.Button(vis_window, text="Close", 
                              command=vis_window.destroy)
        close_btn.pack(pady=10)

# Run the application
if __name__ == "__main__":
    root = tk.Tk()
    app = StyledMovieRecommender(root)
    root.mainloop()