
export class TableInfoService {
  async getTotalRows(): Promise<number> {
    const response = await fetch("https://lancedb--art-search-get-total-rows.modal.run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
    })
    const data = await response.json()
    if (data.status === "error") {
      throw new Error(data.message)
    }
    return data.total_rows || 0
  }
}

export const tableInfoService = new TableInfoService()