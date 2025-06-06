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
    <div className="min-h-screen flex flex-col bg-gray-50">
      <nav className="bg-white border-b border-gray-200 shadow-sm">
        <div className="container mx-auto px-4 py-3 max-w-3xl flex items-center justify-between">
          <Image
            src="/logo.png"
            alt="Wikipedia Search Logo"
            width={100}
            height={100}
            priority
            className="transition-transform hover:scale-105"
          />
          <div className="flex items-center gap-6">
            <a 
              href="https://lancedb.com/pricing" 
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-semibold text-gray-900 hover:text-blue-600 transition-colors duration-200"
            >
              Pricing
            </a>
            <a 
              href="https://accounts.lancedb.com/sign-up" 
              target="_blank"
              rel="noopener noreferrer"
              className="text-sm font-semibold text-gray-900 px-3 py-1.5 border border-gray-300 rounded-md hover:border-blue-600 hover:text-blue-600 transition-all duration-200"
            >
              Sign Up
            </a>
          </div>
        </div>
      </nav>

      <div className="flex-1">
        <div className="container mx-auto px-4 py-8 max-w-3xl">
          <div className="flex flex-col items-center text-center gap-4 mb-8">
            <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-blue-800 bg-clip-text text-transparent">Wikipedia Search</h1>
            <div className="w-16 h-16 transition-transform hover:scale-110">
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
              <div 
                className="relative w-[300px] h-10 bg-gray-100 rounded-full p-0.5 flex items-center"
                onMouseEnter={(e) => {
                  const rect = e.currentTarget.getBoundingClientRect();
                  setTooltip({
                    text: "Click to switch between Semantic, Keyword, or Hybrid search",
                    x: rect.left,
                    y: rect.bottom + 10
                  });
                }}
                onMouseLeave={() => setTooltip(null)}
              >
                <div 
                  className={`absolute h-9 rounded-full transition-all duration-300 ease-in-out
                    ${searchType === 'vector' ? 'w-[98px] left-0.5 bg-blue-600' : 
                      searchType === 'full_text' ? 'w-[98px] left-[100px] bg-blue-600' : 
                      'w-[98px] left-[199.5px] bg-blue-600'}`}
                />
                <div className="relative w-full h-full flex items-center justify-between text-sm font-medium">
                  <button 
                    onClick={() => setSearchType('vector')}
                    className={`z-10 w-[98px] h-full flex items-center justify-center rounded-full transition-all duration-300 ease-in-out text-center
                      ${searchType === 'vector' ? 'text-white' : 'text-gray-600 hover:text-gray-800'}`}
                    onMouseEnter={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      setTooltip({
                        text: "Semantic search understands the meaning behind your query, finding conceptually similar content even if the exact words don't match",
                        x: rect.left,
                        y: rect.bottom + 10
                      });
                    }}
                    onMouseLeave={() => setTooltip(null)}
                  >
                    <span className="w-full text-center">Semantic</span>
                  </button>
                  <button 
                    onClick={() => setSearchType('full_text')}
                    className={`z-10 w-[98px] h-full flex items-center justify-center rounded-full transition-all duration-300 ease-in-out text-center
                      ${searchType === 'full_text' ? 'text-white' : 'text-gray-600 hover:text-gray-800'}`}
                    onMouseEnter={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      setTooltip({
                        text: "Keyword search finds exact matches of your query terms, perfect for specific phrases or technical terms",
                        x: rect.left,
                        y: rect.bottom + 10
                      });
                    }}
                    onMouseLeave={() => setTooltip(null)}
                  >
                    <span className="w-full text-center">Keyword</span>
                  </button>
                  <button 
                    onClick={() => setSearchType('hybrid')}
                    className={`z-10 w-[98px] h-full flex items-center justify-center rounded-full transition-all duration-300 ease-in-out text-center
                      ${searchType === 'hybrid' ? 'text-white' : 'text-gray-600 hover:text-gray-800'}`}
                    onMouseEnter={(e) => {
                      const rect = e.currentTarget.getBoundingClientRect();
                      setTooltip({
                        text: "Hybrid search combines both semantic and keyword approaches for the most comprehensive results",
                        x: rect.left,
                        y: rect.bottom + 10
                      });
                    }}
                    onMouseLeave={() => setTooltip(null)}
                  >
                    <span className="w-full text-center">Hybrid</span>
                  </button>
                </div>
              </div>
            </div>
            <div className="flex gap-2">
        <input
          type="text"
          placeholder="Explore Wikipedia's vast knowledge..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyPress={(e) => e.key === 'Enter' && handleSearch()}
                className="flex-1 p-2 border rounded-lg shadow-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200"
              />
        <button
          onClick={handleSearch}
          disabled={loading}
                className="px-6 py-2 bg-blue-600 text-white rounded-lg disabled:bg-blue-400 hover:bg-blue-700 transition-all duration-200 shadow-sm hover:shadow-md disabled:shadow-none"
        >
          {loading ? 'Searching...' : 'Search'}
        </button>
      </div>

            {tooltip && (
              <div 
                className="fixed z-50 px-3 py-1.5 text-sm text-gray-600 bg-white border border-gray-200 rounded-md shadow-lg max-w-xs transition-all duration-200"
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

          {!results.length && (
            <div className="max-w-2xl mx-auto mt-[100px] mb-[300px] p-8 bg-blue-50 border border-blue-100 rounded-lg shadow-sm">
              <p className="text-gray-600 italic text-sm leading-relaxed space-y-3">
                <span className="block">
                  We&apos;ve uploaded over 40 million Wikipedia documents to LanceDB, enabling you to experience lightning-fast 
                  retrieval in a production environment.
                </span>
                <span className="block">
                  Choose between <span className="font-semibold not-italic">Semantic search</span> for understanding meaning, 
                  <span className="font-semibold not-italic"> Keyword search</span> for exact matches, 
                  or <span className="font-semibold not-italic">Hybrid search</span> that combines both approaches.
                </span>
                <span className="block">
                  Simply type your question and discover relevant information from our vast knowledge base.
                </span>
              </p>
            </div>
          )}

          {(responseTime || backendTime) && (
            <div className="fixed bottom-4 right-4 bg-black/90 text-white px-4 py-2 rounded-lg transition-all duration-300 shadow-lg backdrop-blur-sm">
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
                    className={`transition-all duration-300 ease-in-out ${
                      activeTab === 'results' 
                        ? 'block opacity-100' 
                        : 'hidden opacity-0'
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
                              className="group border border-gray-200 rounded-lg p-3 hover:border-blue-200 hover:shadow-lg transition-all duration-200 bg-white"
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
                              className="px-4 py-2 border rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-50 hover:border-blue-200 hover:text-blue-600 transition-all duration-200 shadow-sm hover:shadow-md disabled:shadow-none"
                            >
                              Previous
                            </button>
                            <div className="flex gap-1">
                              {[...Array(Math.min(4, totalPages))].map((_, i) => (
                                <button
                                  key={i}
                                  onClick={() => setCurrentPage(i + 1)}
                                  className={`px-3 py-1 rounded-lg border transition-all duration-200 ${
                                    currentPage === i + 1
                                      ? 'bg-blue-600 text-white border-blue-600 shadow-md shadow-blue-100'
                                      : 'bg-white text-gray-700 border-gray-200 hover:bg-blue-50 hover:border-blue-200 hover:text-blue-600 hover:shadow-sm'
                                  }`}
                                >
                                  {i + 1}
                                </button>
                              ))}
                              {totalPages > 4 && (
                                <>
                                  <span className="px-2 py-1 text-gray-500">...</span>
                                  <button
                                    onClick={() => setCurrentPage(totalPages)}
                                    className={`px-3 py-1 rounded-lg border transition-all duration-200 ${
                                      currentPage === totalPages
                                        ? 'bg-blue-600 text-white border-blue-600 shadow-md shadow-blue-100'
                                        : 'bg-white text-gray-700 border-gray-200 hover:bg-blue-50 hover:border-blue-200 hover:text-blue-600 hover:shadow-sm'
                                    }`}
                                  >
                                    {totalPages}
                                  </button>
                                </>
                              )}
                            </div>
            <button
                              onClick={() => setCurrentPage(prev => Math.min(prev + 1, totalPages))}
                              disabled={currentPage === totalPages}
                              className="px-4 py-2 border rounded-lg disabled:opacity-50 disabled:cursor-not-allowed hover:bg-blue-50 hover:border-blue-200 hover:text-blue-600 transition-all duration-200 shadow-sm hover:shadow-md disabled:shadow-none"
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
                    className={`transition-all duration-300 ease-in-out ${
                      activeTab === 'parameters' 
                        ? 'block opacity-100' 
                        : 'hidden opacity-0'
                    }`}
                  >
                    {activeTab === 'parameters' && (
                      <div className="bg-white rounded-lg border border-gray-200 shadow-sm mb-[300px]">
                        <div className="p-4">
                          <h3 className="text-sm font-semibold text-gray-800 mb-3 text-center">Search Parameters</h3>
                          <div className="grid grid-cols-2 gap-4">
                            <div className="bg-gray-50 p-3 rounded-lg border border-gray-100 shadow-sm hover:shadow-md transition-all duration-200">
                              <h4 className="text-xs font-semibold text-gray-500 mb-2 pb-1 border-b border-gray-200">Search Configuration</h4>
                              <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                  <span 
                                    className="text-xs text-gray-500 cursor-help"
                                    onMouseEnter={(e) => {
                                      const rect = e.currentTarget.getBoundingClientRect();
                                      setTooltip({
                                        text: "The type of search to perform: Semantic (vector similarity), Keyword (exact matches), or Hybrid (combination of both)",
                                        x: rect.left,
                                        y: rect.bottom + 10
                                      });
                                    }}
                                    onMouseLeave={() => setTooltip(null)}
                                  >
                                    Search Type
                                  </span>
                                  <span className="text-xs font-medium text-blue-600">
                                    {searchType === 'vector' ? 'Semantic' : 
                                     searchType === 'full_text' ? 'Keyword' : 'Hybrid'}
                                  </span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <span 
                                    className="text-xs text-gray-500 cursor-help"
                                    onMouseEnter={(e) => {
                                      const rect = e.currentTarget.getBoundingClientRect();
                                      setTooltip({
                                        text: "The search query or question you want to find information about",
                                        x: rect.left,
                                        y: rect.bottom + 10
                                      });
                                    }}
                                    onMouseLeave={() => setTooltip(null)}
                                  >
                                    Query
                                  </span>
                                  <span className="text-xs font-medium text-gray-800 max-w-[200px] truncate">{query}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <span 
                                    className="text-xs text-gray-500 cursor-help"
                                    onMouseEnter={(e) => {
                                      const rect = e.currentTarget.getBoundingClientRect();
                                      setTooltip({
                                        text: "Maximum number of results to retrieve from the search",
                                        x: rect.left,
                                        y: rect.bottom + 10
                                      });
                                    }}
                                    onMouseLeave={() => setTooltip(null)}
                                  >
                                    Limit
                                  </span>
                                  <span className="text-xs font-medium text-gray-800">100</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <span 
                                    className="text-xs text-gray-500 cursor-help"
                                    onMouseEnter={(e) => {
                                      const rect = e.currentTarget.getBoundingClientRect();
                                      setTooltip({
                                        text: "Whether to include the query execution plan in the response",
                                        x: rect.left,
                                        y: rect.bottom + 10
                                      });
                                    }}
                                    onMouseLeave={() => setTooltip(null)}
                                  >
                                    Explain
                                  </span>
                                  <span className="text-xs font-medium text-gray-800">true</span>
                                </div>
                              </div>
                            </div>
                            <div className="bg-gray-50 p-3 rounded-lg border border-gray-100 shadow-sm hover:shadow-md transition-all duration-200">
                              <h4 className="text-xs font-semibold text-gray-500 mb-2 pb-1 border-b border-gray-200">Results Configuration</h4>
                              <div className="space-y-2">
                                <div className="flex items-center justify-between">
                                  <span 
                                    className="text-xs text-gray-500 cursor-help"
                                    onMouseEnter={(e) => {
                                      const rect = e.currentTarget.getBoundingClientRect();
                                      setTooltip({
                                        text: "Number of results displayed per page in the UI",
                                        x: rect.left,
                                        y: rect.bottom + 10
                                      });
                                    }}
                                    onMouseLeave={() => setTooltip(null)}
                                  >
                                    Results Per Page
                                  </span>
                                  <span className="text-xs font-medium text-gray-800">{resultsPerPage}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <span 
                                    className="text-xs text-gray-500 cursor-help"
                                    onMouseEnter={(e) => {
                                      const rect = e.currentTarget.getBoundingClientRect();
                                      setTooltip({
                                        text: "Total number of results found for the current search",
                                        x: rect.left,
                                        y: rect.bottom + 10
                                      });
                                    }}
                                    onMouseLeave={() => setTooltip(null)}
                                  >
                                    Total Results
                                  </span>
                                  <span className="text-xs font-medium text-gray-800">{results.length}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <span 
                                    className="text-xs text-gray-500 cursor-help"
                                    onMouseEnter={(e) => {
                                      const rect = e.currentTarget.getBoundingClientRect();
                                      setTooltip({
                                        text: "Current page number being displayed",
                                        x: rect.left,
                                        y: rect.bottom + 10
                                      });
                                    }}
                                    onMouseLeave={() => setTooltip(null)}
                                  >
                                    Current Page
                                  </span>
                                  <span className="text-xs font-medium text-gray-800">{currentPage}</span>
                                </div>
                                <div className="flex items-center justify-between">
                                  <span 
                                    className="text-xs text-gray-500 cursor-help"
                                    onMouseEnter={(e) => {
                                      const rect = e.currentTarget.getBoundingClientRect();
                                      setTooltip({
                                        text: "Total number of pages available for the current search results",
                                        x: rect.left,
                                        y: rect.bottom + 10
                                      });
                                    }}
                                    onMouseLeave={() => setTooltip(null)}
                                  >
                                    Total Pages
                                  </span>
                                  <span className="text-xs font-medium text-gray-800">{totalPages}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Query Plan Tab */}
                  <div 
                    className={`transition-all duration-300 ease-in-out ${
                      activeTab === 'queryPlan' 
                        ? 'block opacity-100' 
                        : 'hidden opacity-0'
                    }`}
                  >
                    {activeTab === 'queryPlan' && queryPlan && (
                      <div className="bg-white rounded-lg border border-gray-200 shadow-sm">
                        <div className="p-4">
                          <div className="mb-4">
                            <h3 className="text-sm font-semibold text-gray-800 mb-2 text-center">Query Plan - API Response</h3>
                            <pre className="font-mono text-xs leading-relaxed overflow-x-auto text-gray-800 bg-gray-50 p-2 rounded border border-gray-200 shadow-sm">
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
        </div>
      </div>

      <footer className="bg-white border-t border-gray-200 shadow-sm">
        <div className="container mx-auto px-4 py-6 max-w-3xl">
          <div className="flex flex-col md:flex-row justify-between items-center gap-4">
            <div className="flex items-center gap-2">
              <span className="text-sm text-gray-600">Powered by LanceDB</span>
            </div>
            <div className="flex items-center gap-6">
              <a 
                href="https://lancedb.com/documentation" 
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-600 hover:text-blue-600 transition-colors duration-200"
              >
                Documentation
              </a>
              <a 
                href="https://github.com/lancedb/lancedb" 
                target="_blank"
                rel="noopener noreferrer"
                className="text-sm text-gray-600 hover:text-blue-600 transition-colors duration-200"
              >
                GitHub
              </a>
              <a 
                href="https://discord.gg/G5DcmnZWKB" 
              target="_blank"
              rel="noopener noreferrer"
                className="text-sm text-gray-600 hover:text-blue-600 transition-colors duration-200"
            >
                Discord
            </a>
            </div>
          </div>
        </div>
      </footer>
      </div>
  );
}
