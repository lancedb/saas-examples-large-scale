
class EmbeddingService {
  async searchByText(query: string, searchType: string, limit: number = 20) {
    const response = await fetch("https://lancedb--art-search-search.modal.run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        query, 
        search_type: searchType,
        limit: limit
      })
    });
    const data = await response.json();
    if (data.status === "error") {
      throw new Error(data.message);
    }
    return data.data || [];
  }

  async searchByImage(image: File, searchType: string, limit: number = 20) {
    const base64Image = await this.fileToBase64(image);
    const response = await fetch("https://lancedb--art-search-search.modal.run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        image: base64Image, 
        search_type: searchType,
        limit: limit 
      })
    });
    const data = await response.json();
    if (data.status === "error") {
      throw new Error(data.message);
    }
    return data.data || [];
  }

  async searchRandom(searchType: string = "clip_text_to_image", limit: number = 20) {
    const response = await fetch("https://lancedb--art-search-search.modal.run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ 
        search_type: searchType,
        limit: limit 
      })
    });
    const data = await response.json();
    if (data.status === "error") {
      throw new Error(data.message);
    }
    return data.data || [];
  }

  private fileToBase64(file: File): Promise<string> {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.readAsDataURL(file);
      reader.onload = () => resolve(reader.result as string);
      reader.onerror = error => reject(error);
    });
  }
}

export const embeddingService = new EmbeddingService();