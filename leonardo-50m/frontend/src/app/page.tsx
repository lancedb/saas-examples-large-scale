'use client';

import { useState, useEffect, useCallback } from 'react';
import { SearchResult } from '@/types';
import { API_ENDPOINTS } from '@/config/endpoints';

export default function Home() {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [totalRows, setTotalRows] = useState<number | null>(null);
  const [searchTime, setSearchTime] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showDescription, setShowDescription] = useState(false);

  const fetchTotalRows = useCallback(async () => {
    try {
      const response = await fetch(API_ENDPOINTS.totalRows);
      if (!response.ok) {
        throw new Error(`error! status: ${response.status}`);
      }
      const data = await response.json();
      if (data.status === 'error') {
        throw new Error(data.message || 'Failed to fetch total rows');
      }
      setTotalRows(data.total_rows ?? 0);
    } catch (err) {
      console.error('Error fetching total rows:', err);
      setError(err instanceof Error ? err.message : 'Could not fetch total rows.');
      setTotalRows(0); // Set to 0 on error
    }
  }, []);

  useEffect(() => {
    fetchTotalRows();
  }, [fetchTotalRows]);

  const handleSearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError(null);
    setSearchTime(null);
    setResults([]); 

    try {
      const response = await fetch(API_ENDPOINTS.search, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query.trim(),
          limit: 20,
        }),
      });

      if (!response.ok) {
         const errorData = await response.json().catch(() => ({ message: 'Unknown server error' }));
         throw new Error(errorData.message || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      if (data.status === 'success') {
        setResults(Array.isArray(data.data) ? data.data : []);
        setSearchTime(data.search_time ?? null);
      } else {
        throw new Error(data.message || 'Search failed');
      }
    } catch (err) {
      console.error('Search error:', err);
      setError(err instanceof Error ? err.message : 'An unknown error occurred during search.');
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <main className="container mx-auto px-4 py-8 max-w-7xl">
      <div className="flex flex-col sm:flex-row justify-between items-center mb-8 gap-4">
        <h1 className="text-3xl font-bold text-gray-800">Synthetic Image Search</h1>
        {totalRows !== null && (
          <div className="text-lg text-blue-600 font-medium">
            Total Images: {totalRows.toLocaleString()}
          </div>
        )}
      </div>

      <div className="flex flex-col sm:flex-row gap-3 mb-8">
        <input
          type="text"
          placeholder="Enter search description (e.g., 'a cat wearing a hat')"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          className="flex-grow p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none"
          disabled={loading}
        />
        <button
          onClick={handleSearch}
          disabled={loading || !query.trim()}
          className="px-6 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:bg-gray-400 disabled:cursor-not-allowed transition duration-150 ease-in-out"
        >
          {loading ? (
            <div className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Searching...
            </div>
          ) : (
            'Search'
          )}
        </button>
      </div>

      <div className="mb-4 flex items-center">
        <input
          type="checkbox"
          id="showDescriptionCheckbox"
          checked={showDescription}
          onChange={(e) => setShowDescription(e.target.checked)}
          className="mr-2 h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
        />
        <label htmlFor="showDescriptionCheckbox" className="text-sm text-gray-700">
          Show Prompt
        </label>
      </div>

      {error && (
        <div className="mb-6 p-4 bg-red-100 text-red-700 border border-red-300 rounded-lg">
          <strong>Error:</strong> {error}
        </div>
      )}

      {searchTime !== null && !loading && (
         <div className="mb-6 text-sm text-gray-600">
           LanceDB seach took : {searchTime.toFixed(3)} seconds
         </div>
       )}

      {loading && !results.length && (
         <div className="text-center py-10 text-gray-500">Loading results...</div>
      )}

      {!loading && !results.length && query && !error && (
         <div className="text-center py-10 text-gray-500">No results found for your query.</div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-6">
        {results.map((result, index) => (
          <div key={index} className="border border-gray-200 rounded-lg shadow-sm overflow-hidden bg-white flex flex-col">
            {result.image ? (
              <img
                src={`data:image/jpeg;base64,${result.image}`}
                alt={result.description || 'Search result image'}
                className="result-image" 
                loading="lazy"
              />
            ) : (
              <div className="h-64 bg-gray-100 flex items-center justify-center text-gray-400 rounded-t-lg">
                No Image Available
              </div>
            )}
            <div className="p-4 flex-grow flex flex-col justify-between">
              {showDescription && (
                <p className="text-gray-700 text-sm mb-3 flex-grow">
                  {result.description || 'No description available.'}
                </p>
              )}
              <a
                href={result.url}
                target="_blank"
                rel="noopener noreferrer"
                className="text-xs text-blue-600 hover:underline break-all"
                title={result.url} 
              >
                Source Link
              </a>
            </div>
          </div>
        ))}
      </div>
    </main>
  );
}
