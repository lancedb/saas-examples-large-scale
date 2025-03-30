
class EmbeddingService {

  async searchByText(query: string, modelType: string = "clip", limit: number = 20) {
    const response = await fetch("https://lancedb--art-search-search.modal.run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, model: modelType, limit: limit })
    })
    const data = await response.json()
    if (data.status === "error") {
      throw new Error(data.message)
    }
    return data.data || []
  }

  async searchByImage(image: File, modelType: string = "clip", limit: number = 20) {
    const base64Image = await this.fileToBase64(image)
    const response = await fetch("https://lancedb--art-search-search.modal.run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image: base64Image, model: modelType, limit: limit })
    })
    const data = await response.json()
    if (data.status === "error") {
      throw new Error(data.message)
    }
    return data.data || []
  }

  async searchRandom(modelType: string = "clip", limit: number = 20) {
    const response = await fetch("https://lancedb--art-search-search.modal.run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelType, limit: limit })
      })
      const data = await response.json()
      if (data.status === "error") {
        throw new Error(data.message)
      }
      console.log("recieved data ", data.data)
      return data.data || []
    }
  private fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader()
      reader.readAsDataURL(file)
      reader.onload = () => resolve(reader.result as string)
      reader.onerror = error => reject(error)
    })
  }
}

export const embeddingService = new EmbeddingService()