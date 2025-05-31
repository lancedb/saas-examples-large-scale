'use client';

import { useState, useEffect } from 'react';
import { SearchResult } from '@/types';
import { API_ENDPOINTS } from '@/config/endpoints';
import Image from 'next/image';

export default function Home() {
  const [query, setQuery] = useState('Who wrote Romeo and Juliet?');
  const [searchType, setSearchType] = useState('vector');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [loading, setLoading] = useState(false);
  const [totalRows, setTotalRows] = useState(0);
  const [responseTime, setResponseTime] = useState<string | null>(null);
  const [backendTime, setBackendTime] = useState<string | null>(null);
  const [currentPage, setCurrentPage] = useState(1);
  const resultsPerPage = 3;
  const [activeTab, setActiveTab] = useState('results');

  const [tooltip, setTooltip] = useState<{ text: string; x: number; y: number } | null>(null);

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
            explain: true
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
    <div className="min-h-screen flex flex-col">
      <nav className="bg-white border-b border-gray-200">
        <div className="container mx-auto px-4 py-3 max-w-3xl">
          <Image
            src="/logo.png"
            alt="Wikipedia Search Logo"
            width={80}
            height={80}
            priority
          />
        </div>
      </nav>

      <main className="container mx-auto px-4 py-6 max-w-3xl">
        <div className="flex flex-col items-center text-center gap-4 mb-8">
          <h1 className="text-3xl font-bold">Wikipedia Search</h1>
          <div className="w-16 h-16">
            <Image
              src="/hero.png"
              alt="Globe illustration"
              width={64}
              height={64}
              className="w-full h-full object-contain"
              priority
            />
          </div>
          <div className="text-xl text-blue-600 font-medium">
            Total Documents: {totalRows.toLocaleString()}
          </div>
        </div>

        <div className="flex flex-col gap-3 mb-6">
          <div className="flex justify-center gap-2 mb-2">
            <button
              onClick={() => setSearchType('vector')}
              onMouseEnter={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                setTooltip({
                  text: "Semantic search understands the meaning of your query and finds conceptually similar content, even if the exact words don't match.",
                  x: rect.left,
                  y: rect.bottom + 10
                });
              }}
              onMouseLeave={() => setTooltip(null)}
              className={`px-4 py-2 rounded-lg border transition-colors ${
                searchType === 'vector'
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
              }`}
            >
              Semantic
            </button>
            <button
              onClick={() => setSearchType('full_text')}
              onMouseEnter={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                setTooltip({
                  text: "Keyword search finds exact matches of words in your query within the content.",
                  x: rect.left,
                  y: rect.bottom + 10
                });
              }}
              onMouseLeave={() => setTooltip(null)}
              className={`px-4 py-2 rounded-lg border transition-colors ${
                searchType === 'full_text'
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
              }`}
            >
              Keyword
            </button>
            <button
              onClick={() => setSearchType('hybrid')}
              onMouseEnter={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                setTooltip({
                  text: "Hybrid search combines both semantic and keyword search to provide the most relevant results.",
                  x: rect.left,
                  y: rect.bottom + 10
                });
              }}
              onMouseLeave={() => setTooltip(null)}
              className={`px-4 py-2 rounded-lg border transition-colors ${
                searchType === 'hybrid'
                  ? 'bg-blue-600 text-white border-blue-600'
                  : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
              }`}
            >
              Hybrid
            </button>
          </div>
          <div className="flex gap-2">
            <input
              type="text"
              placeholder="Enter your fact to check"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
              className="flex-1 p-2 border rounded-lg"
            />
            <button
              onClick={handleSearch}
              disabled={loading}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg disabled:bg-blue-400 hover:bg-blue-700 transition-colors"
            >
              {loading ? 'Searching...' : 'Search'}
            </button>
          </div>

          {tooltip && (
            <div 
              className="fixed z-50 px-3 py-1.5 text-sm text-gray-600 bg-white border border-gray-200 rounded-md shadow-sm max-w-xs transition-opacity duration-200"
              style={{
                left: tooltip.x,
                top: tooltip.y,
                transform: 'translateX(-50%)'
              }}
            >
              {tooltip.text}
            </div>
          )}
        </div>

        {(responseTime || backendTime) && (
          <div className="fixed bottom-4 right-4 bg-black text-white px-4 py-2 rounded-lg transition-opacity">
            {backendTime && <div>{backendTime}</div>}
            {responseTime && <div>{responseTime}</div>}
          </div>
        )}

        {results.length > 0 && (
          <div className="mt-8">
            {/* Tab Navigation */}
            <div className="border-b border-gray-200">
              <nav className="flex -mb-px">
                <button
                  onClick={() => setActiveTab('results')}
                  className={`px-4 py-2 text-sm font-medium border-b-2 transition-all duration-200 ease-in-out ${
                    activeTab === 'results'
                      ? 'border-blue-600 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Search Results
                </button>
                <button
                  onClick={() => setActiveTab('parameters')}
                  className={`px-4 py-2 text-sm font-medium border-b-2 transition-all duration-200 ease-in-out ${
                    activeTab === 'parameters'
                      ? 'border-blue-600 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Search Parameters
                </button>
                <button
                  onClick={() => setActiveTab('queryPlan')}
                  className={`px-4 py-2 text-sm font-medium border-b-2 transition-all duration-200 ease-in-out ${
                    activeTab === 'queryPlan'
                      ? 'border-blue-600 text-blue-600'
                      : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
                  }`}
                >
                  Query Plan
                </button>
              </nav>
            </div>

            {/* Tab Content */}
            <div className="mt-4">
              <div className="relative">
                {/* Search Results Tab */}
                <div 
                  className={`absolute w-full transition-all duration-300 ease-in-out ${
                    activeTab === 'results' 
                      ? 'opacity-100 translate-x-0' 
                      : 'opacity-0 -translate-x-4 pointer-events-none'
                  }`}
                >
                  {activeTab === 'results' && (
                    <div className="space-y-4">
                      <div className="flex items-center mb-3">
                        <h2 className="text-lg font-semibold text-gray-800">
                          Search Results <span className="text-gray-500 font-normal">(showing {startIndex + 1}-{Math.min(endIndex, results.length)} of {results.length})</span>
                        </h2>
                      </div>
                      <div className="space-y-2">
                        {currentResults.map((result, index) => (
                          <div 
                            key={index} 
                            className="group border border-gray-200 rounded-lg p-3 hover:border-blue-200 hover:shadow-md transition-all duration-200 bg-white"
                          >
                            <h2 className="text-base font-semibold mb-1.5 text-gray-800 group-hover:text-blue-600 transition-colors">
                              {result.title}
                            </h2>
                            <p className="text-gray-600 text-sm leading-relaxed mb-2 overflow-hidden" style={{ 
                              display: '-webkit-box',
                              WebkitLineClamp: 2,
                              WebkitBoxOrient: 'vertical'
                            }}>
                              {result.content}
                            </p>
                            <a
                              href={result.url}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-flex items-center text-sm text-blue-600 hover:text-blue-800 transition-colors"
                            >
                              <svg 
                                className="w-3.5 h-3.5 mr-1" 
                                fill="none" 
                                stroke="currentColor" 
                                viewBox="0 0 24 24"
                              >
                                <path 
                                  strokeLinecap="round" 
                                  strokeLinejoin="round" 
                                  strokeWidth={2} 
                                  d="M10 6H6a2 2 0 00-2 2v10a2 2 0 002 2h10a2 2 0 002-2v-4M14 4h6m0 0v6m0-6L10 14" 
                                />
                              </svg>
                              View on Wikipedia
                            </a>
                          </div>
                        ))}
                      </div>

                      {/* Pagination Controls */}
                      <div className="flex justify-end items-center mt-6">
                        <div className="flex items-center gap-2">
                          <button
                            onClick={() => setCurrentPage(prev => Math.max(prev - 1, 1))}
                            disabled={currentPage === 1}
                            className="px-4 py-2 border rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-gray-50"
                          >
                            Previous
                          </button>
                          <div className="flex gap-1">
                            {[...Array(Math.min(4, totalPages))].map((_, i) => (
                              <button
                                key={i}
                                onClick={() => setCurrentPage(i + 1)}
                                className={`px-3 py-1 rounded-lg border ${
                                  currentPage === i + 1
                                    ? 'bg-blue-600 text-white border-blue-600'
                                    : 'bg-white text-gray-700 border-gray-200 hover:bg-gray-50'
                                }`}
                              >
                                {i + 1}
                              </button>
                            ))}
                            {totalPages > 4 && (
                              <span className="px-2 py-1 text-gray-500">...</span>
                            )}
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
                    </div>
                  )}
                </div>

                {/* Search Parameters Tab */}
                <div 
                  className={`absolute w-full transition-all duration-300 ease-in-out ${
                    activeTab === 'parameters' 
                      ? 'opacity-100 translate-x-0' 
                      : 'opacity-0 translate-x-4 pointer-events-none'
                  }`}
                >
                  {activeTab === 'parameters' && (
                    <div className="bg-white rounded-lg border border-gray-200">
                      <div className="p-4">
                        <div className="grid grid-cols-2 gap-4 text-sm">
                          <div>
                            <span className="text-gray-500">Search Type:</span>
                            <span className="ml-2 text-gray-800 font-medium">
                              {searchType === 'vector' ? 'Semantic' : 
                               searchType === 'full_text' ? 'Keyword' : 'Hybrid'}
                            </span>
                          </div>
                          <div>
                            <span className="text-gray-500">Query:</span>
                            <span className="ml-2 text-gray-800 font-medium">{query}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Limit:</span>
                            <span className="ml-2 text-gray-800 font-medium">100</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Explain:</span>
                            <span className="ml-2 text-gray-800 font-medium">true</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Results Per Page:</span>
                            <span className="ml-2 text-gray-800 font-medium">{resultsPerPage}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Total Results:</span>
                            <span className="ml-2 text-gray-800 font-medium">{results.length}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Current Page:</span>
                            <span className="ml-2 text-gray-800 font-medium">{currentPage}</span>
                          </div>
                          <div>
                            <span className="text-gray-500">Total Pages:</span>
                            <span className="ml-2 text-gray-800 font-medium">{totalPages}</span>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                {/* Query Plan Tab */}
                <div 
                  className={`absolute w-full transition-all duration-300 ease-in-out ${
                    activeTab === 'queryPlan' 
                      ? 'opacity-100 translate-x-0' 
                      : 'opacity-0 translate-x-4 pointer-events-none'
                  }`}
                >
                  {activeTab === 'queryPlan' && queryPlan && (
                    <div className="bg-white rounded-lg border border-gray-200">
                      <div className="p-4">
                        <div className="mb-4">
                          <h3 className="text-sm font-semibold text-gray-800 mb-2">Query Plan - API Response</h3>
                          <pre className="font-mono text-xs leading-relaxed overflow-x-auto text-gray-800 bg-gray-50 p-2 rounded border border-gray-200">
                            {queryPlan?.split('\n').map((line, i) => (
                              <div key={i} className="whitespace-pre">
                                {line}
                              </div>
                            ))}
                          </pre>
                        </div>
                        <div>
                          <h3 className="text-sm font-semibold text-gray-800 mb-1 text-center">Query Plan Flow</h3>
                          <p className="text-xs text-gray-500 mb-2 text-center">(in order of execution, from bottom to top)</p>
                          <div className="space-y-0 max-w-md mx-auto">
                            <div className="bg-gray-50 rounded-t p-1.5 border border-gray-200">
                              <h4 className="font-medium text-blue-600 text-xs mb-0.5 text-center">1. QuantizedIvfExec</h4>
                              <p className="text-gray-600 text-xs leading-tight text-center">
                                Uses a quantized Inverted File (IVF) index for efficient vector similarity search.
                              </p>
                            </div>
                            <div className="flex justify-center">
                              <div className="relative w-0.5 h-3 bg-gray-300">
                                <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[3px] border-l-transparent border-r-[3px] border-r-transparent border-t-[3px] border-t-gray-300"></div>
                              </div>
                            </div>
                            <div className="bg-gray-50 p-1.5 border-x border-gray-200">
                              <h4 className="font-medium text-blue-600 text-xs mb-0.5 text-center">2. ANNSubIndex</h4>
                              <p className="text-gray-600 text-xs leading-tight text-center">
                                Performs Approximate Nearest Neighbor search using the vector index, retrieving the 100 most similar vectors.
                              </p>
                            </div>
                            <div className="flex justify-center">
                              <div className="relative w-0.5 h-3 bg-gray-300">
                                <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[3px] border-l-transparent border-r-[3px] border-r-transparent border-t-[3px] border-t-gray-300"></div>
                              </div>
                            </div>
                            <div className="bg-gray-50 p-1.5 border-x border-gray-200">
                              <h4 className="font-medium text-blue-600 text-xs mb-0.5 text-center">3. SortExec</h4>
                              <p className="text-gray-600 text-xs leading-tight text-center">
                                Sorts the results by similarity score (_distance) in ascending order, keeping only the top 100 results.
                              </p>
                            </div>
                            <div className="flex justify-center">
                              <div className="relative w-0.5 h-3 bg-gray-300">
                                <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[3px] border-l-transparent border-r-[3px] border-r-transparent border-t-[3px] border-t-gray-300"></div>
                              </div>
                            </div>
                            <div className="bg-gray-50 p-1.5 border-x border-gray-200">
                              <h4 className="font-medium text-blue-600 text-xs mb-0.5 text-center">4. GlobalLimitExec</h4>
                              <p className="text-gray-600 text-xs leading-tight text-center">
                                Limits the total number of results to 100, starting from the first result (skip=0).
                              </p>
                            </div>
                            <div className="flex justify-center">
                              <div className="relative w-0.5 h-3 bg-gray-300">
                                <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[3px] border-l-transparent border-r-[3px] border-r-transparent border-t-[3px] border-t-gray-300"></div>
                              </div>
                            </div>
                            <div className="bg-gray-50 p-1.5 border-x border-gray-200">
                              <h4 className="font-medium text-blue-600 text-xs mb-0.5 text-center">5. CoalesceBatchesExec</h4>
                              <p className="text-gray-600 text-xs leading-tight text-center">
                                Combines multiple data batches into a single batch of size 1024 for efficient processing.
                              </p>
                            </div>
                            <div className="flex justify-center">
                              <div className="relative w-0.5 h-3 bg-gray-300">
                                <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[3px] border-l-transparent border-r-[3px] border-r-transparent border-t-[3px] border-t-gray-300"></div>
                              </div>
                            </div>
                            <div className="bg-gray-50 p-1.5 border-x border-gray-200">
                              <h4 className="font-medium text-blue-600 text-xs mb-0.5 text-center">6. RemoteTake</h4>
                              <p className="text-gray-600 text-xs leading-tight text-center">
                                Retrieves the specified columns from the remote data source.
                              </p>
                            </div>
                            <div className="flex justify-center">
                              <div className="relative w-0.5 h-3 bg-gray-300">
                                <div className="absolute -bottom-1 left-1/2 transform -translate-x-1/2 w-0 h-0 border-l-[3px] border-l-transparent border-r-[3px] border-r-transparent border-t-[3px] border-t-gray-300"></div>
                              </div>
                            </div>
                            <div className="bg-gray-50 rounded-b p-1.5 border border-gray-200">
                              <h4 className="font-medium text-blue-600 text-xs mb-0.5 text-center">7. ProjectionExec</h4>
                              <p className="text-gray-600 text-xs leading-tight text-center">
                                Selects and renames specific columns from the results: content, title, url, identifier, chunk_index, and _distance (similarity score).
                              </p>
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
