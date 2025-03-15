import { useState, useEffect } from 'react'
import { embeddingService } from './services/embeddingService'
import { tableInfoService } from './services/getTableInfo'
import './App.css'

interface Artwork {
  object_id: number
  title: string
  artist: string
  date: string
  medium: string
  department: string
  culture: string
  img_url: string
}

function App() {
  const [searchQuery, setSearchQuery] = useState('')
  const [artworks, setArtworks] = useState<Artwork[]>([])
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [numImgs, setNumImgs] = useState<number>(20)
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
      const response = await embeddingService.searchRandom("clip", numImgs)
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
    try {
      let results
      if (selectedImage) {
        console.log("searching by image")
        results = await embeddingService.searchByImage(selectedImage)
      } else if (searchQuery.trim()) {
        console.log("searching text")
        results = await embeddingService.searchByText(searchQuery)
      } else {
        console.log("searching random")
        results = await fetchRandomArtworks()
      }
      setArtworks(results)
    } catch (error) {
      console.error('Error searching artworks:', error)
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
        </div>
      </header>

      <main className="gallery">
        {artworks.length === 0 ? (
          <p>No artworks to display. Try searching for something!</p>
        ) : (
          artworks.map((artwork) => (
            <div key={artwork.object_id} className="artwork-card">
              <img 
                src={artwork.img_url} 
                alt={artwork.title} 
              />
              <div className="artwork-info">
                <h3>{artwork.title}</h3>
                <p>{artwork.artist}</p>
                <p>{artwork.date}</p>
                <p>{artwork.department}</p>
              </div>
            </div>
          ))
        )}
      </main>
    </div>
  )
}

export default App
