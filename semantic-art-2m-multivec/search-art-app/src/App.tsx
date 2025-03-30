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

function App() {
  const [searchQuery, setSearchQuery] = useState('')
  const [artworks, setArtworks] = useState<Artwork[]>([])
  const [searchMessage, setSearchMessage] = useState<string | null>(null)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [numImgs, setNumImgs] = useState<number>(20)
  const [modelType, setModelType] = useState<'clip' | 'siglip'>('clip')
  const [totalImgs, setTotalImgs] = useState<number | null>(null)
  const [previewImage, setPreviewImage] = useState<string | null>(null)

  useEffect(() => {
    const fetchInitialArtworks = async () => {
      console.log("searching random")
      const results = await fetchRandomArtworks()
      setArtworks(results)
      console.log("random", results)
    }
    tableInfoService.getTotalRows().then((total) => {
      setTotalImgs(total)
    })
    fetchInitialArtworks()
  }, [])

  const fetchRandomArtworks = async () => {
    try {
      const response = await embeddingService.searchRandom(modelType, numImgs)
      return response
    } catch (error) {
      console.error('Error fetching random artworks:', error)
    }
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

  const handleSearch = async () => {
    setLoading(true)
    const startTime = performance.now()
    try {
      let results
      if (selectedImage) {
        console.log("searching by image")
        results = await embeddingService.searchByImage(selectedImage, modelType, numImgs)
      } else if (searchQuery.trim()) {
        console.log("searching text")
        results = await embeddingService.searchByText(searchQuery, modelType, numImgs)
      } else {
        console.log("searching random")
        results = await fetchRandomArtworks()
      }
      const endTime = performance.now()
      const searchTime = ((endTime - startTime) / 1000).toFixed(2)
      setSearchMessage(`Search completed in ${searchTime} seconds`)
      setTimeout(() => setSearchMessage(null), 3000) // Message disappears after 3 seconds
      setArtworks(results)
    } catch (error) {
      console.error('Error searching artworks:', error)
      setSearchMessage('Error occurred during search')
      setTimeout(() => setSearchMessage(null), 3000)
    } finally {
      setLoading(false)
    }
  }

  // Use a spinner while loading the page if there are no artworks to display
  if (artworks.length === 0) {
    return <div className='container'><div className="spinner"></div></div>
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
          <button onClick={handleSearch} disabled={loading}>
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
              value={modelType} 
              onChange={(e) => setModelType(e.target.value as 'clip' | 'siglip')}
              className="model-select"
            >
              <option value="clip">CLIP</option>
              <option value="siglip">SigLIP</option>
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
