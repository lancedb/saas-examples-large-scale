'use client';

import { useState, useEffect } from 'react';
import { SearchResult } from '@/types';
import { API_ENDPOINTS } from '@/config/endpoints';

export default function Home() {
  const [query, setQuery] = useState('');
  const [searchType, setSearchType] = useState('vector');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [totalRows, setTotalRows] = useState(0);
  const [responseTime, setResponseTime] = useState<string | null>(null);
  const [backendTime, setBackendTime] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const resultsPerPage = 5;

  useEffect(() => {
    fetchTotalRows();
  }, []);

  const fetchTotalRows = async () => {
    try {
      const response = await fetch(API_ENDPOINTS.totalRows, {
        method: 'GET',
        headers: {
          'Accept': 'application/json',
        },
      });
      const data = await response.json();
      setTotalRows(data.total_rows || 0);
    } catch (error) {
      console.error('Error fetching total rows:', error);
      setTotalRows(0);
    }
  };

  const [showQueryPlan, setShowQueryPlan] = useState(false);
  const [queryPlan, setQueryPlan] = useState<string | null>(null);
  const [explainEnabled, setExplainEnabled] = useState(false);

  const handleSearch = async () => {
      if (!query.trim()) return;
  
      setLoading(true);
      setCurrentPage(1); // Reset to first page on new search
      const startTime = performance.now();
  
      try {
        const response = await fetch(API_ENDPOINTS.search, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query,
            search_type: searchType,
            limit: 100,
            explain: explainEnabled
          }),
        });
  
        const data = await response.json();
        const endTime = performance.now();
        const frontendTime = ((endTime - startTime) / 1000).toFixed(2);
        
        console.log('Search response:', data);
        setResponseTime(`Rendering took: ${frontendTime}s`);
        setBackendTime(`Search took: ${data.search_time.toFixed(2)}s`);
        setQueryPlan(data.query_plan);
        console.log('Query plan:', data.query_plan);
  
        if (data.status === 'success') {
          setResults(Array.isArray(data.data) ? data.data : []);
        } else {
          console.error('Search error:', data.message);
          setResults([]);
        }
  
        setTimeout(() => {
          setResponseTime(null);
          setBackendTime(null);
        }, 3000);
      } catch (error) {
        console.error('Search error:', error);
        setResults([]);
      } finally {
        setLoading(false);
      }
  };

  // Calculate pagination values
  const totalPages = Math.ceil(results.length / resultsPerPage);
  const startIndex = (currentPage - 1) * resultsPerPage;
  const endIndex = startIndex + resultsPerPage;
  const currentResults = results.slice(startIndex, endIndex);

  return (
    <main className="container mx-auto px-4 py-8 max-w-6xl">
      <div className="flex justify-between items-center mb-8">
        <h1 className="text-3xl font-bold">Fact Check Wiki</h1>
        <div className="text-xl text-blue-600">
          Total Documents: {totalRows.toLocaleString()}
        </div>
      </div>

      <div className="flex gap-4 mb-8">
        <input
          type="text"
          placeholder="Enter your fact to check"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
          className="flex-1 p-2 border rounded-lg"
        />
        <select
          value={searchType}
          onChange={(e) => setSearchType(e.target.value)}
          className="p-2 border rounded-lg w-48"
        >
          <option value="vector">Vector Search</option>
          <option value="full_text">Full Text Search</option>
          <option value="hybrid">Hybrid Search</option>
        </select>
        <button
          onClick={handleSearch}
          disabled={loading}
          className="px-6 py-2 bg-blue-600 text-white rounded-lg disabled:bg-blue-400"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

      <div className="flex items-center gap-2 mb-4">
        <label className="flex items-center gap-2 text-sm">
          <input
            type="checkbox"
            checked={explainEnabled}
            onChange={(e) => setExplainEnabled(e.target.checked)}
            className="form-checkbox h-4 w-4 text-blue-600"
          />
          Show Query Plan
        </label>
      </div>

      {queryPlan && explainEnabled && (
        <div className="mb-8 border border-gray-200 rounded-lg">
          <div className="flex justify-between items-center p-4 bg-gray-50 border-b border-gray-200">
            <h3 className="font-semibold text-gray-700">Query Plan</h3>
            <button
              onClick={() => setShowQueryPlan(!showQueryPlan)}
              className="text-blue-600 hover:text-blue-800 font-medium"
            >
              {showQueryPlan ? 'Hide' : 'Show'}
            </button>
          </div>
          {showQueryPlan && (
            <pre className="p-6 bg-white font-mono text-sm leading-relaxed overflow-x-auto text-gray-800">
              {queryPlan?.split('\n').map((line, i) => (
                <div key={i} className="whitespace-pre">
                  {line}
                </div>
              ))}
            </pre>
          )}
        </div>
      )}

      {(responseTime || backendTime) && (
        <div className="fixed bottom-4 right-4 bg-black text-white px-4 py-2 rounded-lg transition-opacity">
          {backendTime && <div>{backendTime}</div>}
          {responseTime && <div>{responseTime}</div>}
        </div>
      )}

      <div className="space-y-4">
        {currentResults.map((result, index) => (
          <div key={index} className="border rounded-lg p-4 shadow-sm">
            <h2 className="text-xl font-semibold mb-2">{result.title}</h2>
            <p className="text-gray-700 mb-2">{result.content}</p>
            <a
              href={result.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-600 hover:underline"
            >
              Source
            </a>
          </div>
        ))}

        {/* Pagination Controls */}
        {results.length > 0 && (
          <div className="flex justify-between items-center mt-6">
            <div className="text-sm text-gray-600">
              Showing {startIndex + 1}-{Math.min(endIndex, results.length)} of {results.length} results
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                disabled={currentPage === 1}
                className="px-4 py-2 border rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
              >
                Previous
              </button>
              <div className="px-4 py-2 text-gray-600">
                Page {currentPage} of {totalPages}
              </div>
              <button
                onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                disabled={currentPage === totalPages}
                className="px-4 py-2 border rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
              >
                Next
              </button>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
