import React, { useState, useEffect, useRef } from 'react';
import { 
  MessageSquare, 
  Send, 
  Sparkles, 
  RefreshCw, 
  ShoppingBag, 
  Layers, 
  DollarSign, 
  Sliders, 
  Check, 
  AlertCircle,
  HelpCircle,
  TrendingUp,
  Star,
  ArrowRight,
  User,
  Bot
} from 'lucide-react';
import axios from 'axios';

export default function App() {
  // System State
  const [loading, setLoading] = useState(false);
  const [initializing, setInitializing] = useState(false);
  const [status, setStatus] = useState({
    status: 'checking',
    database_ready: false,
    faiss_ready: false,
    product_count: 0,
    gemini_enabled: false
  });
  
  // Chat State
  const [query, setQuery] = useState('');
  const [history, setHistory] = useState([
    {
      role: 'assistant',
      content: "Hello! I'm your Virtual Sales Associate today. Tell me what you're looking for, or describe your lifestyle need, and I'll find you the perfect product. (e.g. 'I joined the gym' or 'Laptops for Computer Engineering under 80000')"
    }
  ]);
  
  // UI Tabs & Interactive State
  const [activeTab, setActiveTab] = useState('matches'); // 'matches', 'compare', 'explore'
  const [recommendedProducts, setRecommendedProducts] = useState([]);
  const [comparison, setComparison] = useState(null);
  const [clarifications, setClarifications] = useState([]);
  const [categories, setCategories] = useState([]);
  const [selectedCategory, setSelectedCategory] = useState('');
  const [exploreProducts, setExploreProducts] = useState([]);
  const [exploreLoading, setExploreLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');

  const chatEndRef = useRef(null);

  // Suggested starter queries
  const starterQueries = [
    { text: "Suggest black shirts for Goa trip", cat: "Fashion" },
    { text: "Best camera phone under ₹40000", cat: "Smartphones" },
    { text: "Compare iPhone 16 Pro and Samsung S25 Ultra", cat: "Smartphones" },
    { text: "Laptop for Computer Engineering", cat: "Laptops" },
    { text: "Gym products for beginners", cat: "Fitness" },
    { text: "Best sunscreen for Indian summer", cat: "Skincare" }
  ];

  // Fetch status on startup
  useEffect(() => {
    fetchStatus();
    fetchCategories();
  }, []);

  // Auto scroll chat
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [history]);

  const fetchStatus = async () => {
    try {
      const res = await axios.get('/api/');
      setStatus(res.data);
      if (res.data.product_count > 0) {
        fetchExploreProducts(res.data.product_count);
      }
    } catch (err) {
      console.error("Error fetching system status:", err);
      setStatus(prev => ({ ...prev, status: 'offline' }));
    }
  };

  const fetchCategories = async () => {
    try {
      const res = await axios.get('/api/categories');
      setCategories(res.data.categories || []);
    } catch (err) {
      console.error("Error fetching categories:", err);
    }
  };

  const fetchExploreProducts = async (cnt = 3000) => {
    setExploreLoading(true);
    try {
      const res = await axios.get(`/api/products?top_k=24${selectedCategory ? '&category=' + selectedCategory : ''}`);
      setExploreProducts(res.data.products || []);
    } catch (err) {
      console.error("Error fetching explorer products:", err);
    } finally {
      setExploreLoading(false);
    }
  };

  // Initialize DB triggering
  const handleInitialize = async () => {
    setInitializing(true);
    setErrorMsg('');
    try {
      const res = await axios.post('/api/initialize');
      if (res.data.status === 'success') {
        await fetchStatus();
        await fetchCategories();
        setHistory(prev => [
          ...prev,
          {
            role: 'assistant',
            content: `🎉 Database initialized successfully! Loaded ${res.data.product_count} premium products across Smartphones, Laptops, Skincare, Shoes, Fitness, Fashion, and Accessories. FAISS indexing is complete. How can I help you shop today?`
          }
        ]);
      }
    } catch (err) {
      setErrorMsg(err.response?.data?.detail || "Initialization failed. Check console.");
    } finally {
      setInitializing(false);
    }
  };

  // Chat message submission
  const handleSend = async (messageText) => {
    const textToSend = messageText || query;
    if (!textToSend.trim()) return;

    setErrorMsg('');
    setLoading(true);
    setQuery('');
    
    // Add user message to history
    const updatedHistory = [...history, { role: 'user', content: textToSend }];
    setHistory(updatedHistory);

    try {
      const res = await axios.post('/api/chat', {
        query: textToSend,
        history: updatedHistory.slice(0, -1) // Send context excluding the new query
      });

      // Add assistant response to history
      setHistory(prev => [...prev, { role: 'assistant', content: res.data.response }]);
      
      // Update recommendations & comparison data
      if (res.data.products && res.data.products.length > 0) {
        setRecommendedProducts(res.data.products);
        setActiveTab('matches');
      }
      
      if (res.data.comparison) {
        setComparison(res.data.comparison);
        setActiveTab('compare');
      } else if (res.data.intent !== 'comparison') {
        // If not a comparison intent, reset comparison view to keep it clean
        setComparison(null);
      }
      
      setClarifications(res.data.clarification_questions || []);

    } catch (err) {
      console.error("Chat error:", err);
      setErrorMsg("Failed to get response from assistant. Make sure backend is running.");
      setHistory(prev => [
        ...prev,
        { role: 'assistant', content: "⚠️ Sorry, I encountered an error communicating with my database. Please check my server status." }
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleSuggestionClick = (queryText) => {
    handleSend(queryText);
  };

  const handleCategoryFilterChange = (catName) => {
    setSelectedCategory(catName);
  };

  useEffect(() => {
    fetchExploreProducts();
  }, [selectedCategory]);

  return (
    <div className="min-h-screen lg:h-screen lg:max-h-screen flex flex-col font-sans select-none lg:overflow-hidden">
      
      {/* Top Banner Status Bar */}
      <header className="glass-card border-b border-white/5 py-4 px-6 sticky top-0 z-40 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-accent-500 flex items-center justify-center shadow-lg shadow-brand-500/25">
            <ShoppingBag className="w-5 h-5 text-white" />
          </div>
          <div>
            <h1 className="text-xl font-bold tracking-tight text-white flex items-center gap-2">
              AmzSales <span className="gradient-text font-extrabold uppercase text-xs tracking-wider border border-brand-500/30 px-2 py-0.5 rounded-full">Virtual Associate</span>
            </h1>
            <p className="text-xs text-slate-400">AI-Powered E-commerce Chatbot & Retrieval Engine</p>
          </div>
        </div>

        <div className="flex items-center gap-4">
          {/* Status Badge */}
          <div className="flex items-center gap-6 text-xs border-r border-white/10 pr-6 mr-2 hidden md:flex">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${status.database_ready ? 'bg-emerald-500' : 'bg-amber-500'}`} />
              <span className="text-slate-300">Catalog: <strong className="text-white">{status.product_count} items</strong></span>
            </div>
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${status.gemini_enabled ? 'bg-emerald-500' : 'bg-red-500'}`} />
              <span className="text-slate-300">Gemini LLM: <strong className="text-white">{status.gemini_enabled ? 'Active' : 'Offline Mode'}</strong></span>
            </div>
          </div>

          {/* Database Setup Button */}
          {!status.database_ready ? (
            <button 
              onClick={handleInitialize} 
              disabled={initializing}
              className="px-4 py-2 rounded-lg bg-gradient-to-r from-brand-600 to-brand-500 hover:from-brand-500 hover:to-brand-400 text-white text-xs font-semibold flex items-center gap-2 shadow-lg shadow-brand-500/25 interactive-button"
            >
              {initializing ? <RefreshCw className="w-3.5 h-3.5 animate-spin" /> : <Sparkles className="w-3.5 h-3.5" />}
              {initializing ? 'Generating 3000+ Products...' : 'Initialize Demo Database (3000+ Products)'}
            </button>
          ) : (
            <button 
              onClick={handleInitialize}
              disabled={initializing}
              className="px-3 py-1.5 rounded-lg border border-white/10 hover:bg-white/5 text-slate-300 text-xs font-medium flex items-center gap-2 interactive-button"
              title="Regenerate & Reindex FAISS database"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${initializing ? 'animate-spin' : ''}`} />
              Re-Index FAISS
            </button>
          )}
        </div>
      </header>

      {/* Main Split Screen Area */}
      <main className="flex-1 flex flex-col lg:flex-row lg:overflow-hidden max-w-[1800px] w-full mx-auto">
        
        {/* Left Side: Conversational Chat Pane */}
        <section className="w-full lg:w-[48%] lg:h-full flex flex-col border-b lg:border-b-0 lg:border-r border-white/5 bg-brand-950/20 backdrop-blur-sm min-h-[500px] lg:min-h-0">
          
          {/* Chat Messages */}
          <div className="flex-1 overflow-y-auto p-6 space-y-4 custom-scrollbar">
            {history.map((msg, idx) => (
              <div 
                key={idx} 
                className={`flex gap-3.5 max-w-[85%] ${msg.role === 'user' ? 'ml-auto flex-row-reverse' : 'mr-auto'}`}
              >
                {/* Avatar */}
                <div className={`w-9 h-9 rounded-lg flex items-center justify-center shadow-md flex-shrink-0
                  ${msg.role === 'user' 
                    ? 'bg-gradient-to-tr from-brand-500 to-indigo-500' 
                    : 'bg-brand-900 border border-brand-500/20'
                  }`}
                >
                  {msg.role === 'user' ? <User className="w-4.5 h-4.5 text-white" /> : <Bot className="w-4.5 h-4.5 text-brand-300" />}
                </div>

                {/* Message Bubble */}
                <div className={`rounded-2xl p-4 text-sm leading-relaxed shadow-sm
                  ${msg.role === 'user' 
                    ? 'bg-brand-600 text-white rounded-tr-none' 
                    : 'glass-card border border-white/5 text-slate-200 rounded-tl-none'
                  }`}
                >
                  {/* Handle newline formatting in responses */}
                  <div className="whitespace-pre-line space-y-2">
                    {msg.content}
                  </div>
                </div>
              </div>
            ))}
            
            {/* Thinking Loader */}
            {loading && (
              <div className="flex gap-3.5 mr-auto">
                <div className="w-9 h-9 rounded-lg bg-brand-900 border border-brand-500/20 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4.5 h-4.5 text-brand-300 animate-pulse" />
                </div>
                <div className="glass-card border border-white/5 text-slate-300 rounded-2xl rounded-tl-none p-4 flex items-center gap-1.5">
                  <span className="w-2 h-2 rounded-full bg-brand-400 animate-bounce" style={{ animationDelay: '0ms' }}></span>
                  <span className="w-2 h-2 rounded-full bg-brand-400 animate-bounce" style={{ animationDelay: '150ms' }}></span>
                  <span className="w-2 h-2 rounded-full bg-brand-400 animate-bounce" style={{ animationDelay: '300ms' }}></span>
                </div>
              </div>
            )}
            
            <div ref={chatEndRef} />
          </div>

          {/* Interactive Assistant Clarification Chips & Quick Prompts */}
          <div className="px-6 py-3 border-t border-white/5 bg-brand-950/40">
            {clarifications.length > 0 ? (
              <div>
                <div className="text-[11px] font-bold text-brand-300 mb-2 uppercase tracking-widest flex items-center gap-1">
                  <HelpCircle className="w-3 h-3" /> Assistant Follow-up Clarification
                </div>
                <div className="flex flex-wrap gap-2">
                  {clarifications.map((qText, qIdx) => (
                    <button 
                      key={qIdx}
                      onClick={() => handleSuggestionClick(qText)}
                      className="px-3 py-1.5 rounded-full glass-card border border-brand-500/30 hover:border-brand-500 bg-brand-900/40 hover:bg-brand-800/50 text-brand-200 hover:text-white text-xs transition-colors duration-200 text-left flex items-center gap-1.5"
                    >
                      <span>{qText}</span>
                      <ArrowRight className="w-3 h-3 text-brand-400 flex-shrink-0" />
                    </button>
                  ))}
                </div>
              </div>
            ) : (
              <div>
                <div className="text-[11px] font-bold text-slate-400 mb-2 uppercase tracking-widest flex items-center gap-1">
                  <TrendingUp className="w-3 h-3 text-brand-400" /> Vague Query Test Prompts
                </div>
                <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
                  {starterQueries.map((item, qIdx) => (
                    <button 
                      key={qIdx}
                      onClick={() => handleSuggestionClick(item.text)}
                      className="p-2 text-left rounded-lg glass-card border border-white/5 hover:border-brand-500/30 bg-brand-950/50 text-[11px] text-slate-300 hover:text-white transition-colors duration-200 truncate"
                      title={item.text}
                    >
                      <span className="text-brand-400 text-[10px] block font-semibold uppercase">{item.cat}</span>
                      {item.text}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>

          {/* Chat Form Input */}
          <form 
            onSubmit={(e) => { e.preventDefault(); handleSend(); }}
            className="p-4 border-t border-white/5 bg-brand-950/80 flex items-center gap-2"
          >
            <input 
              type="text" 
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder={status.database_ready ? "Ask your sales associate anything..." : "Please initialize database first..."}
              disabled={loading || !status.database_ready}
              className="flex-1 bg-brand-950 border border-white/10 focus:border-brand-500 focus:ring-1 focus:ring-brand-500 outline-none rounded-xl px-4 py-3 text-sm text-white placeholder-slate-500 disabled:opacity-40"
            />
            <button 
              type="submit"
              disabled={loading || !status.database_ready || !query.trim()}
              className="w-12 h-12 rounded-xl bg-gradient-to-tr from-brand-600 to-indigo-600 hover:from-brand-500 hover:to-indigo-500 flex items-center justify-center text-white transition-colors shadow-md shadow-brand-500/10 interactive-button"
            >
              <Send className="w-4.5 h-4.5" />
            </button>
          </form>

          {/* System Info / Warnings */}
          {errorMsg && (
            <div className="bg-red-950/50 border-t border-red-500/20 text-red-300 text-xs px-6 py-2.5 flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-400 flex-shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}
          
        </section>

        {/* Right Side: Product Catalog Hub (Tabs) */}
        <section className="w-full lg:w-[52%] lg:h-full flex flex-col bg-brand-950/10">
          
          {/* Tabs Header */}
          <div className="flex border-b border-white/5 bg-brand-950/40">
            <button 
              onClick={() => setActiveTab('matches')}
              className={`flex-1 py-4 text-sm font-semibold flex items-center justify-center gap-2 border-b-2 transition-all
                ${activeTab === 'matches' 
                  ? 'border-brand-500 text-white bg-white/5' 
                  : 'border-transparent text-slate-400 hover:text-slate-200'}`}
            >
              <Sparkles className="w-4 h-4" /> Assisted Matches
              {recommendedProducts.length > 0 && (
                <span className="ml-1 text-[10px] bg-brand-500 text-white px-2 py-0.5 rounded-full">{recommendedProducts.length}</span>
              )}
            </button>
            <button 
              onClick={() => setActiveTab('compare')}
              className={`flex-1 py-4 text-sm font-semibold flex items-center justify-center gap-2 border-b-2 transition-all
                ${activeTab === 'compare' 
                  ? 'border-brand-500 text-white bg-white/5' 
                  : 'border-transparent text-slate-400 hover:text-slate-200'}`}
            >
              <Layers className="w-4 h-4" /> Product Comparer
              {comparison && (
                <span className="ml-1 text-[10px] bg-accent-500 text-white px-2 py-0.5 rounded-full">Ready</span>
              )}
            </button>
            <button 
              onClick={() => setActiveTab('explore')}
              className={`flex-1 py-4 text-sm font-semibold flex items-center justify-center gap-2 border-b-2 transition-all
                ${activeTab === 'explore' 
                  ? 'border-brand-500 text-white bg-white/5' 
                  : 'border-transparent text-slate-400 hover:text-slate-200'}`}
            >
              <Sliders className="w-4 h-4" /> Shop Explorer
            </button>
          </div>

          {/* Tab Contents */}
          <div className="flex-1 overflow-y-auto p-6 custom-scrollbar">
            
            {/* TAB 1: ASSISTED MATCHES */}
            {activeTab === 'matches' && (
              <div className="space-y-6">
                {recommendedProducts.length === 0 ? (
                  <div className="text-center py-24 glass-card rounded-2xl border border-white/5 p-8 max-w-md mx-auto mt-12">
                    <Sparkles className="w-12 h-12 text-brand-400 mx-auto mb-4 pulse-slow" />
                    <h3 className="text-base font-bold text-white mb-2">No recommendations active</h3>
                    <p className="text-xs text-slate-400">Ask the associate assistant to suggest some products, and their ranked matches will instantly populate here.</p>
                  </div>
                ) : (
                  <div>
                    <h2 className="text-sm font-bold text-brand-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                      <Check className="w-4 h-4 text-brand-400" /> Ranked Matches (FAISS Vector Results)
                    </h2>
                    
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                      {recommendedProducts.map((p, idx) => (
                        <div 
                          key={p.product_id}
                          className="glass-card glass-card-hover rounded-xl overflow-hidden border border-white/5 flex flex-col relative"
                        >
                          {/* Rank Badge */}
                          <div className="absolute top-3 left-3 w-6 h-6 rounded-full bg-brand-950/80 border border-brand-500/30 flex items-center justify-center text-[11px] font-bold text-brand-300 z-10">
                            #{idx + 1}
                          </div>

                          {/* Image linked to Flipkart with category name */}
                          <a href={p.product_url} target="_blank" rel="noopener noreferrer" className="block relative h-44 bg-brand-950 flex items-center justify-center overflow-hidden border-b border-white/5 group">
                            <div className="absolute inset-0 bg-cover bg-center filter brightness-[0.4] blur-[3px]" style={{ backgroundImage: `url(${p.image_url})` }}></div>
                            <img src={p.image_url} alt={p.product_title} className="h-full object-contain relative z-10 py-3 transition-transform duration-300 group-hover:scale-105" />
                            <span className="absolute top-3 right-3 z-20 text-[9px] font-bold tracking-wider px-2 py-0.5 rounded bg-black/60 text-slate-300 border border-white/10 uppercase">
                              {p.category}
                            </span>
                          </a>

                          {/* Info */}
                          <div className="p-4 flex-1 flex flex-col justify-between">
                            <div>
                              <div className="flex justify-between items-start gap-2 mb-1.5">
                                <span className="text-[10px] text-brand-400 font-bold uppercase">{p.brand}</span>
                                <div className="flex items-center text-[11px] text-amber-400 font-semibold gap-0.5">
                                  <Star className="w-3 h-3 fill-current" />
                                  <span>{p.rating} ({p.review_count})</span>
                                </div>
                              </div>
                              <a href={p.product_url} target="_blank" rel="noopener noreferrer" className="hover:text-brand-300 transition-colors block">
                                <h4 className="text-xs font-bold text-white line-clamp-2 hover:line-clamp-none transition-all mb-3" title={p.product_title}>
                                  {p.product_title}
                                </h4>
                              </a>
                            </div>

                            <div>
                              <div className="flex items-baseline gap-1.5 mb-3">
                                <span className="text-base font-extrabold text-white">₹{p.selling_price.toLocaleString('en-IN')}</span>
                              </div>

                              {/* Spec mini chips */}
                              <div className="space-y-1.5 border-t border-white/5 pt-3 mb-3 text-[11px] text-slate-300">
                                {p.category === 'Smartphones' && (
                                  <>
                                    <div>⚡ {p.processor} | 🔋 {p.battery}</div>
                                    <div>💾 {p.ram} RAM / {p.storage} Storage</div>
                                    <div>📷 Camera: {p.camera}</div>
                                  </>
                                )}
                                {p.category === 'Laptops' && (
                                  <>
                                    <div>🖥️ {p.processor} | 🎮 {p.gpu}</div>
                                    <div>💾 {p.ram} RAM / {p.storage}</div>
                                    <div>🔋 Backup: {p.battery_backup}</div>
                                  </>
                                )}
                                {p.category === 'Fashion' && (
                                  <>
                                    <div>🧵 {p.material} | 👕 Fit: {p.fit}</div>
                                    <div>🎨 Color: {p.color} | Style: {p.style}</div>
                                  </>
                                )}
                                {p.category === 'Shoes' && (
                                  <>
                                    <div>👟 Type: {p.type} | 🧵 {p.material}</div>
                                    <div>☁️ {p.comfort_level}</div>
                                  </>
                                )}
                                {p.category === 'Skincare' && (
                                  <>
                                    <div>🧴 Skin: {p.skin_type} | 🛡️ {p.spf}</div>
                                    <div>🧪 Ingr: {p.ingredients}</div>
                                  </>
                                )}
                                {p.category === 'Fitness' && (
                                  <>
                                    <div>🏋️ {p.product_type} | {p.weight}</div>
                                    <div>💪 Usage: {p.usage}</div>
                                  </>
                                )}
                                {p.category === 'Accessories' && (
                                  <>
                                    <div>🎧 Type: {p.type} | {p.material}</div>
                                    <div>📱 Compat: {p.compatibility}</div>
                                  </>
                                )}
                              </div>

                              {/* Relevance scores details */}
                              <div className="bg-brand-950/60 rounded-lg p-2 text-[10px] space-y-1 border border-white/5">
                                <div className="text-slate-400 font-bold uppercase tracking-wider text-[8px]">Ranking Factors:</div>
                                <div className="flex justify-between text-slate-300">
                                  <span>Semantic Match:</span>
                                  <span className="font-semibold text-brand-300">{Math.round(p.semantic_score * 100)}%</span>
                                </div>
                                <div className="flex justify-between text-slate-300">
                                  <span>Popularity (Reviews):</span>
                                  <span className="font-semibold text-brand-300">{Math.round(p.popularity_score * 100)}%</span>
                                </div>
                                <div className="flex justify-between text-slate-300">
                                  <span>Price Relevance:</span>
                                  <span className="font-semibold text-emerald-400">{Math.round(p.price_relevance * 100)}%</span>
                                </div>
                                <div className="flex justify-between font-bold text-white border-t border-white/5 pt-1 mt-1 text-[11px]">
                                  <span>Final Weighted Score:</span>
                                  <span className="text-indigo-400">{p.final_ranking_score.toFixed(4)}</span>
                                </div>
                              </div>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 2: PRODUCT COMPARER */}
            {activeTab === 'compare' && (
              <div className="space-y-6">
                {!comparison ? (
                  <div className="text-center py-24 glass-card rounded-2xl border border-white/5 p-8 max-w-md mx-auto mt-12">
                    <Layers className="w-12 h-12 text-accent-500 mx-auto mb-4 pulse-slow" />
                    <h3 className="text-base font-bold text-white mb-2">No products loaded for comparison</h3>
                    <p className="text-xs text-slate-400">Ask the associate to compare models (e.g. "Compare iPhone 16 Pro and Samsung S25 Ultra") to generate a specifications matrix.</p>
                  </div>
                ) : (
                  <div>
                    <h2 className="text-sm font-bold text-accent-500 uppercase tracking-widest mb-4 flex items-center gap-2">
                      <Layers className="w-4 h-4" /> Side-By-Side Product Comparison
                    </h2>

                    <div className="overflow-x-auto glass-card rounded-xl border border-white/5">
                      <table className="w-full text-left border-collapse text-xs">
                        <thead>
                          <tr className="border-b border-white/10 bg-brand-900/60">
                            {comparison.headers.map((hdr, hIdx) => (
                              <th 
                                key={hIdx} 
                                className="p-3.5 font-extrabold text-white uppercase tracking-wider"
                                style={{ minWidth: hIdx === 0 ? '110px' : '150px' }}
                              >
                                {hdr}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {comparison.rows.map((row, rIdx) => {
                            const isAspectHeader = row[0] === 'Key Pros' || row[0] === 'Key Cons';
                            return (
                              <tr 
                                key={rIdx} 
                                className={`border-b border-white/5 hover:bg-white/5 transition-all
                                  ${isAspectHeader ? 'bg-brand-950/40 font-semibold' : ''}`}
                              >
                                {row.map((cell, cIdx) => (
                                  <td 
                                    key={cIdx} 
                                    className={`p-3.5 leading-relaxed
                                      ${cIdx === 0 ? 'font-bold text-slate-400 border-r border-white/5' : 'text-slate-200'}
                                      ${row[0] === 'Key Pros' && cIdx > 0 ? 'text-emerald-400' : ''}
                                      ${row[0] === 'Key Cons' && cIdx > 0 ? 'text-rose-400' : ''}
                                    `}
                                  >
                                    {cell}
                                  </td>
                                ))}
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* TAB 3: EXPLORER / DIRECT FILTERS */}
            {activeTab === 'explore' && (
              <div className="space-y-6">
                <div>
                  <h2 className="text-sm font-bold text-brand-300 uppercase tracking-widest mb-4 flex items-center gap-2">
                    <Sliders className="w-4 h-4 text-brand-400" /> Filter Shop Database
                  </h2>
                  
                  {/* Category Chips filter selector */}
                  <div className="flex flex-wrap gap-2 mb-6">
                    <button
                      onClick={() => handleCategoryFilterChange('')}
                      className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all
                        ${!selectedCategory 
                          ? 'bg-brand-500 border-brand-500 text-white shadow-md' 
                          : 'glass-card border-white/5 text-slate-400 hover:text-slate-200'}`}
                    >
                      All Stock
                    </button>
                    {categories.map((cat) => (
                      <button
                        key={cat.name}
                        onClick={() => handleCategoryFilterChange(cat.name)}
                        className={`px-3 py-1.5 rounded-full text-xs font-semibold border transition-all
                          ${selectedCategory === cat.name 
                            ? 'bg-brand-500 border-brand-500 text-white shadow-md' 
                            : 'glass-card border-white/5 text-slate-400 hover:text-slate-200'}`}
                      >
                        {cat.name} ({cat.count})
                      </button>
                    ))}
                  </div>

                  {/* Explore Grid */}
                  {exploreLoading ? (
                    <div className="text-center py-24">
                      <RefreshCw className="w-10 h-10 text-brand-500 animate-spin mx-auto mb-4" />
                      <p className="text-sm text-slate-400">Loading catalog items...</p>
                    </div>
                  ) : exploreProducts.length === 0 ? (
                    <div className="text-center py-24 glass-card rounded-2xl border border-white/5">
                      <AlertCircle className="w-12 h-12 text-slate-500 mx-auto mb-4" />
                      <h3 className="text-base font-bold text-white mb-2">Stock database empty</h3>
                      <p className="text-xs text-slate-400">Initialize the demo products at the top to populate items.</p>
                    </div>
                  ) : (
                    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4">
                      {exploreProducts.map((p) => (
                        <div 
                          key={p.product_id}
                          className="glass-card glass-card-hover rounded-xl overflow-hidden border border-white/5 flex flex-col"
                        >
                          <a href={p.product_url} target="_blank" rel="noopener noreferrer" className="block relative h-28 bg-brand-950 flex items-center justify-center overflow-hidden border-b border-white/5 group">
                            <div className="absolute inset-0 bg-cover bg-center filter brightness-[0.4] blur-[2px]" style={{ backgroundImage: `url(${p.image_url})` }}></div>
                            <img src={p.image_url} alt={p.product_title} className="h-full object-contain relative z-10 py-2 transition-transform duration-300 group-hover:scale-105" />
                            <span className="absolute top-2 right-2 z-20 text-[8px] font-extrabold tracking-wider px-2 py-0.5 rounded bg-black/60 text-slate-300 uppercase">
                              {p.category}
                            </span>
                          </a>
                          
                          <div className="p-3 flex-1 flex flex-col justify-between">
                            <div>
                              <div className="flex justify-between items-center text-[10px] text-slate-400 mb-1">
                                <span className="font-bold uppercase text-brand-400">{p.brand}</span>
                                <div className="flex items-center text-amber-400 gap-0.5">
                                  <Star className="w-2.5 h-2.5 fill-current" />
                                  <span>{p.rating}</span>
                                </div>
                              </div>
                              <a href={p.product_url} target="_blank" rel="noopener noreferrer" className="hover:text-brand-300 transition-colors block">
                                <h4 className="text-[11px] font-bold text-white line-clamp-2 hover:line-clamp-none transition-all mb-2" title={p.product_title}>
                                  {p.product_title}
                                </h4>
                              </a>
                            </div>
                            
                            <div className="flex justify-between items-center mt-2 pt-2 border-t border-white/5">
                              <span className="text-xs font-extrabold text-white">₹{p.selling_price.toLocaleString('en-IN')}</span>
                              <button 
                                onClick={() => handleSend(`Show details for ${p.product_id}`)}
                                className="text-[10px] font-bold text-brand-400 hover:text-brand-300 transition-colors uppercase tracking-wider flex items-center gap-0.5"
                              >
                                View Specs
                              </button>
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}

                </div>
              </div>
            )}

          </div>

        </section>

      </main>

    </div>
  );
}
