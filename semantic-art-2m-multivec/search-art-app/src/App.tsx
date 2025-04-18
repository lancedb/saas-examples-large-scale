import { useState, useEffect } from 'react'
import { embeddingService } from './services/embeddingService'
import { tableInfoService } from './services/getTableInfo'
import './App.css'

interface Artwork {
  title: string
  artist: string
  date: string
  image: string
  ds_name: string
  _rowid: number
}

const SEARCH_TYPES = [
  { value: "clip_text_to_image", label: "CLIP: Text to Image Search" },
  { value: "sigplip_text_to_image", label: "SigLIP: Text to Image Search" },
  { value: "clip_image_to_image", label: "CLIP: Image to Image Search" },
  { value: "sigplip_image_to_image", label: "SigLIP: Image to Image Search" },
  { value: "clip_caption", label: "CLIP: Caption Search" },
  { value: "sigplip_caption", label: "SigLIP: Caption Search" },
  { value: "full_text", label: "Full-text Search (Title, Artist, Caption)" },
  { value: "hybrid_clip", label: "CLIP: Hybrid Search" },
  { value: "hybrid_siglip", label: "SigLIP: Hybrid Search" }
];

const TEXT_ONLY_SEARCH_TYPES = ['clip_text_to_image', 'sigplip_text_to_image', 'clip_caption', 'siglip_caption', 'hybrid_clip', 'hybrid_siglip', 'full_text'];
const IMAGE_ONLY_SEARCH_TYPES = ['clip_image_to_image', 'sigplip_image_to_image'];

function App() {
  const [searchType, setSearchType] = useState<string>("clip_text_to_image")
  const [searchQuery, setSearchQuery] = useState('')
  const [artworks, setArtworks] = useState<Artwork[]>([])
  const [searchMessage, setSearchMessage] = useState<string | null>(null)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [numImgs, setNumImgs] = useState<number>(20)
  // Update the model type state to include all search types
  const [totalImgs, setTotalImgs] = useState<number | null>(null)
  const [previewImage, setPreviewImage] = useState<string | null>(null)

  useEffect(() => {
    const fetchInitialArtworks = async () => {
      try {
        const results = await fetchRandomArtworks()
        if (results) {  // Add null check
          setArtworks(results)
        } else {
          console.error("No results received from fetchRandomArtworks")
          setSearchMessage("Error loading initial artworks")
        }
      } catch (error) {
        console.error("Error in fetchInitialArtworks:", error)
        setSearchMessage("Error loading initial artworks")
      }
    }

    const fetchTotalRows = async () => {
      try {
        const total = await tableInfoService.getTotalRows()
        setTotalImgs(total)
      } catch (error) {
        console.error("Error fetching total rows:", error)
      }
    }

    fetchInitialArtworks()
    fetchTotalRows()
  }, [])

  const fetchRandomArtworks = async () => {
    try {
      console.log("Fetching random artworks with model:", searchType)
      const response = await embeddingService.searchByText("random", "clip_text_to_image", numImgs)
      return response
    } catch (error) {
      throw error  // Propagate error to caller
    }
  }

  // Modify the loading state check
  if (!artworks || artworks.length === 0) {
    return (
      <div className='container'>
        <div className="spinner"></div>
        {searchMessage && <div className="search-message">{searchMessage}</div>}
      </div>
    )
  }
  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedImage(file)
      const reader = new FileReader()
      reader.onload = (e) => {
        setPreviewImage(e.target?.result as string)
      }
      reader.readAsDataURL(file)
    }
  }

  const isSearchDisabled = () => {
    // Disable if both text and image are provided
    console.log("searchType ", searchType)
    if (searchQuery.trim() && selectedImage) {
      console.log("Both text and image provided")
      return true;
    }
    // Disable if image is selected but search type is text-only
    if (selectedImage && TEXT_ONLY_SEARCH_TYPES.includes(searchType)) {
      console.log("Image selected but search type is text-only")
      return true;
    }
    // Disable if text is entered but search type is image-only
    if (searchQuery.trim() && IMAGE_ONLY_SEARCH_TYPES.includes(searchType)) {
      console.log("Text entered but search type is image-only")
      return true;
    }
    // Disable if no input is provided (except for random search)
    if (!searchQuery.trim() && !selectedImage) {

      return false; // Allow random search
    }
    return false;
  };

  const handleSearch = async () => {
    setLoading(true);
    const startTime = performance.now();
    console.log("search for type ", searchType )
    try {
      let results;
      if (selectedImage && IMAGE_ONLY_SEARCH_TYPES.includes(searchType)) {
        console.log("searching by image");
        results = await embeddingService.searchByImage(selectedImage, searchType, numImgs);
      } else if (searchQuery.trim() && TEXT_ONLY_SEARCH_TYPES.includes(searchType)) {
        console.log("searching text");
        results = await embeddingService.searchByText(searchQuery, searchType, numImgs);
      } else {
        console.log("searching random");
        results = await embeddingService.searchRandom(searchType, numImgs);
      }
      const endTime = performance.now();
      const searchTime = ((endTime - startTime) / 1000).toFixed(2);
      setSearchMessage(`Search completed in ${searchTime} seconds`);
      setTimeout(() => setSearchMessage(null), 3000);
      setArtworks(results);
    } catch (error) {
      console.error('Error searching artworks:', error);
      setSearchMessage('Error occurred during search');
      setTimeout(() => setSearchMessage(null), 3000);
    } finally {
      setLoading(false);
    }
  };



  // Show image upload only for image-to-image search types
  {searchType.includes("image_to_image") && (
    <input
      type="file"
      accept="image/*"
      onChange={handleImageUpload}
      id="image-upload"
    />
  )}


  // Modify the loading state check
  if (!artworks || artworks.length === 0) {
    return (
      <div className='container'>
        <div className="spinner"></div>
        {searchMessage && <div className="search-message">{searchMessage}</div>}
      </div>
    )
  }

     
  return (
    <div className="container">
      {totalImgs && <div className="total-count">{totalImgs} artworks indexed</div>}
      {searchMessage && <div className="search-message">{searchMessage}</div>}
      <header>
        <div className="search-container">
          {previewImage && (
            <div className="preview-image-container">
              <img src={previewImage} alt="Preview" className="preview-image" />
              <button
                onClick={() => {
                  setSelectedImage(null)
                  setPreviewImage(null)
                }}
                className="clear-image"
              >
                ×
              </button>
            </div>
          )}
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search artworks..."
            onKeyPress={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault()
                handleSearch()
              }
            }}
          />
          <button 
            onClick={handleSearch} 
            disabled={loading || isSearchDisabled()}
            className={`search-button ${isSearchDisabled() ? 'disabled' : ''}`}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
          </button>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            id="image-upload"
          />
          <label htmlFor="image-upload" className="upload-button">
            <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect>
              <circle cx="8.5" cy="8.5" r="1.5"></circle>
              <polyline points="21 15 16 10 5 21"></polyline>
            </svg>
          </label>
          
          {/* Add model type selector and number of images control */}
          <div className="search-controls">
            <select 
              className="model-select"
              value={searchType}
              onChange={(e) => setSearchType(e.target.value)}
            >
              {SEARCH_TYPES.map(type => (
                <option key={type.value} value={type.value}>
                  {type.label}
                </option>
              ))}
            </select>
            <input
              type="number"
              value={numImgs}
              onChange={(e) => setNumImgs(Math.max(1, Math.min(50, parseInt(e.target.value) || 1)))}
              min="1"
              max="100"
              className="num-images-input"
            />
          </div>
        </div>
      </header>

      <main className="gallery">
        {artworks.length === 0 ? (
          <p>No artworks to display. Try searching for something!</p>
        ) : (
          artworks.map((artwork, index) => (
            <div key={`${artwork.ds_name}-${artwork.title}-${index}`} className="artwork-card">
              <img 
                src={`data:image/jpeg;base64,${artwork.image}`}
                alt={artwork.title} 
              />
              <div className="artwork-info">
                <h3> Title: {artwork.title}</h3>
                <p> Artist: {artwork.artist}</p>
                <p> Date: {artwork.date}</p>
                <p> Datset: {artwork.ds_name}</p>
                <p> Row ID: {artwork._rowid}</p>
              </div>
            </div>
          ))
        )}
      </main>
    </div>
  )
}

export default App
