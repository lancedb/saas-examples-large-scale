import { useState, useEffect } from 'react'
import axios from 'axios'
import './App.css'

interface Artwork {
  object_id: number
  title: string
  artist: string
  date: string
  medium: string
  department: string
  culture: string
  img: string
}

function App() {
  const [searchQuery, setSearchQuery] = useState('')
  const [artworks, setArtworks] = useState<Artwork[]>([])
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    // Load random artworks on component mount
    fetchRandomArtworks()
  }, [])

  const fetchRandomArtworks = async () => {
    try {
      const response = await axios.get('http://localhost:8000/random')
      setArtworks(response.data)
    } catch (error) {
      console.error('Error fetching random artworks:', error)
    }
  }

  const handleSearch = async () => {
    if (!searchQuery.trim()) {
      fetchRandomArtworks()
      return
    }

    setLoading(true)
    try {
      const response = await axios.post('http://localhost:8000/search', {
        query: searchQuery
      })
      setArtworks(response.data)
    } catch (error) {
      console.error('Error searching artworks:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      setSelectedImage(file)
      setLoading(true)
      
      try {
        const formData = new FormData()
        formData.append('file', file)
        
        const response = await axios.post(
          'http://localhost:8000/search/image',
          formData,
          {
            headers: {
              'Content-Type': 'multipart/form-data'
            }
          }
        )
        setArtworks(response.data)
      } catch (error) {
        console.error('Error searching by image:', error)
      } finally {
        setLoading(false)
      }
    }
  }

  return (
    <div className="container">
      <header>
        <h1>Met Art Explorer</h1>
        <div className="search-container">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search artworks..."
          />
          <button onClick={handleSearch} disabled={loading}>
            {loading ? 'Searching...' : 'Search'}
          </button>
          <input
            type="file"
            accept="image/*"
            onChange={handleImageUpload}
            id="image-upload"
          />
          <label htmlFor="image-upload" className="upload-button">
            Upload Image for Visual Search
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
                src={'data:image/jpeg;base64,' + artwork.img} 
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
