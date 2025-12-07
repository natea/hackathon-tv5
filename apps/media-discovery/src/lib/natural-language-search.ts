/**
 * Natural Language Search Service
 * Converts user prompts into semantic search queries using AI
 */

import { generateObject, generateText } from 'ai';
import { openai } from '@ai-sdk/openai';
import { google } from '@ai-sdk/google';
import { z } from 'zod';
import type { SemanticSearchQuery, SearchIntent, SearchFilters, MediaContent, SearchResult } from '@/types/media';
import { searchMulti, getSimilarMovies, getSimilarTVShows, discoverMovies, discoverTVShows } from './tmdb';
import { searchByEmbedding, getContentEmbedding, calculateSimilarity } from './vector-search';

// Schema for parsed search intent
// Note: All fields are optional AND nullable to handle AI returning null for empty values
const SearchIntentSchema = z.object({
  mood: z.array(z.string()).optional().nullable().describe('Emotional tone or feeling (e.g., "exciting", "heartwarming", "dark")'),
  themes: z.array(z.string()).optional().nullable().describe('Story themes (e.g., "redemption", "coming-of-age", "survival")'),
  pacing: z.enum(['slow', 'medium', 'fast']).optional().nullable().describe('Preferred pacing of the content'),
  era: z.union([z.string(), z.array(z.string())]).optional().nullable().describe('Time period setting (e.g., "1980s", "modern", "futuristic")'),
  setting: z.array(z.string()).optional().nullable().describe('Physical setting (e.g., "space", "urban", "underwater")'),
  similar_to: z.array(z.string()).optional().nullable().describe('Similar movies/shows mentioned by user'),
  avoid: z.array(z.string()).optional().nullable().describe('Elements to avoid (e.g., "gore", "jump scares")'),
  genres: z.array(z.string()).optional().nullable().describe('Inferred genres from the query'),
  keywords: z.array(z.string()).optional().nullable().describe('Key search terms extracted'),
  mediaType: z.enum(['movie', 'tv', 'all']).optional().nullable().describe('Preferred media type'),
});

const SearchFiltersSchema = z.object({
  mediaType: z.enum(['movie', 'tv', 'all']).optional(),
  genres: z.array(z.number()).optional(),
  yearRange: z.object({
    min: z.number().optional(),
    max: z.number().optional(),
  }).optional(),
  ratingMin: z.number().optional(),
});

// Server-side intent cache (survives across requests in same process)
const intentCache = new Map<string, { result: SemanticSearchQuery; timestamp: number }>();
const INTENT_CACHE_TTL_MS = 10 * 60 * 1000; // 10 minute TTL

// Genre mapping for common genre names to TMDB IDs
const GENRE_MAP: Record<string, { movie: number; tv: number }> = {
  action: { movie: 28, tv: 10759 },
  'action & adventure': { movie: 28, tv: 10759 },
  adventure: { movie: 12, tv: 10759 },
  animation: { movie: 16, tv: 16 },
  animated: { movie: 16, tv: 16 },
  anime: { movie: 16, tv: 16 },  // Anime maps to Animation genre
  cartoon: { movie: 16, tv: 16 },
  comedy: { movie: 35, tv: 35 },
  crime: { movie: 80, tv: 80 },
  documentary: { movie: 99, tv: 99 },
  drama: { movie: 18, tv: 18 },
  family: { movie: 10751, tv: 10751 },
  // Aliases for family-friendly content
  children: { movie: 10751, tv: 10751 },
  "children's": { movie: 10751, tv: 10751 },
  kids: { movie: 10751, tv: 10751 },
  "kid's": { movie: 10751, tv: 10751 },
  "kid-friendly": { movie: 10751, tv: 10751 },
  "family-friendly": { movie: 10751, tv: 10751 },
  fantasy: { movie: 14, tv: 10765 },
  'sci-fi & fantasy': { movie: 14, tv: 10765 },
  history: { movie: 36, tv: 36 },
  historical: { movie: 36, tv: 36 },
  horror: { movie: 27, tv: 27 },
  scary: { movie: 27, tv: 27 },
  music: { movie: 10402, tv: 10402 },
  musical: { movie: 10402, tv: 10402 },
  mystery: { movie: 9648, tv: 9648 },
  romance: { movie: 10749, tv: 10749 },
  romantic: { movie: 10749, tv: 10749 },
  'romantic comedy': { movie: 35, tv: 35 },
  'rom-com': { movie: 35, tv: 35 },
  'sci-fi': { movie: 878, tv: 10765 },
  'science fiction': { movie: 878, tv: 10765 },
  thriller: { movie: 53, tv: 53 },
  suspense: { movie: 53, tv: 53 },
  war: { movie: 10752, tv: 10768 },
  'war & politics': { movie: 10752, tv: 10768 },
  western: { movie: 37, tv: 37 },
};

/**
 * Parse natural language query into structured search intent
 */
export async function parseSearchQuery(query: string): Promise<SemanticSearchQuery> {
  // Normalize for cache key
  const cacheKey = query.toLowerCase().trim();

  // Check cache first
  const cached = intentCache.get(cacheKey);
  if (cached && Date.now() - cached.timestamp < INTENT_CACHE_TTL_MS) {
    console.log(`📦 Intent cache hit for: "${query.slice(0, 30)}..."`);
    return cached.result;
  }

  try {
    console.log(`🧠 AI parsing intent for: "${query.slice(0, 30)}..."`);
    // Use AI to extract intent from natural language
    const { object: intent } = await generateObject({
      model: openai('gpt-4o-mini'),
      schema: SearchIntentSchema,
      prompt: `Analyze this movie/TV show search query and extract the user's intent:

Query: "${query}"

Extract:
- Mood: What emotional experience are they seeking?
- Themes: What story themes are they interested in?
- Pacing: Do they want something slow-burn, fast-paced, or in between?
- Era: What time period setting do they prefer?
- Setting: Where should the story take place?
- Similar to: Are they referencing specific movies/shows?
- Avoid: What elements should be avoided?
- Genres: What genres fit this description?
- Keywords: What are the key search terms?
- Media type: Do they specifically want movies or TV shows?

Be specific and extract as much relevant information as possible.`,
    });

    // Convert genre names to IDs (handle null arrays from AI)
    const genresList = intent.genres || [];
    const genreIds = genresList.flatMap(genre => {
      const normalized = genre.toLowerCase().trim();
      const mapping = GENRE_MAP[normalized];
      if (mapping) {
        return [mapping.movie, mapping.tv].filter((v, i, a) => a.indexOf(v) === i);
      }
      // Log unmapped genres for debugging
      console.log(`⚠️ Unmapped genre: "${genre}" (normalized: "${normalized}")`);
      return [];
    });

    console.log(`🎬 Parsed intent - genres: [${genresList.join(', ')}] -> IDs: [${genreIds.join(', ')}]`);

    const result: SemanticSearchQuery = {
      query,
      intent: intent as SearchIntent,
      filters: {
        mediaType: intent.mediaType || 'all',
        genres: genreIds.length > 0 ? genreIds : undefined,
      },
    };

    // Cache the result
    intentCache.set(cacheKey, { result, timestamp: Date.now() });

    return result;
  } catch (error) {
    console.error('Failed to parse search query with AI:', error);
    // Fallback to basic query
    return { query };
  }
}

/**
 * Perform semantic search combining TMDB and vector search
 */
export async function semanticSearch(
  query: string,
  userPreferences?: number[]
): Promise<SearchResult[]> {
  // Parse the natural language query
  const semanticQuery = await parseSearchQuery(query);
  console.log(`🔍 Semantic query:`, JSON.stringify(semanticQuery, null, 2));

  // Parallel search strategies
  const [tmdbResults, vectorResults] = await Promise.all([
    performTMDBSearch(semanticQuery),
    performVectorSearch(semanticQuery),
  ]);

  console.log(`📊 TMDB results: ${tmdbResults.length}, Vector results: ${vectorResults.length}`);

  // Merge and rank results
  const mergedResults = mergeAndRankResults(tmdbResults, vectorResults, semanticQuery, userPreferences);

  console.log(`📊 Merged results: ${mergedResults.length}`);

  return mergedResults;
}

/**
 * Perform TMDB-based search with intent understanding
 */
async function performTMDBSearch(query: SemanticSearchQuery): Promise<SearchResult[]> {
  const results: SearchResult[] = [];

  // Text search - prioritize direct title matches
  if (query.query) {
    console.log(`🔎 TMDB text search for: "${query.query}"`);
    const { results: textResults } = await searchMulti(query.query, query.filters);
    console.log(`🔎 TMDB text search returned: ${textResults.length} results`);

    // Check if the query looks like a specific title (contains proper nouns or is in similar_to)
    const queryLower = query.query.toLowerCase();
    const isLikelyTitleSearch = query.intent?.similar_to?.some(
      ref => ref.toLowerCase() === queryLower || queryLower.includes(ref.toLowerCase())
    );

    results.push(...textResults.map((content, index) => {
      // Give highest score to exact/close title matches
      const titleLower = content.title.toLowerCase();
      const isExactMatch = titleLower === queryLower;
      const isCloseMatch = titleLower.includes(queryLower) || queryLower.includes(titleLower);

      let score = 0.8;
      if (isExactMatch) {
        score = 1.0; // Perfect match
      } else if (isCloseMatch) {
        score = 0.95; // Close match
      } else if (index < 3) {
        score = 0.85; // Top TMDB results
      }

      return {
        content,
        relevanceScore: score,
        matchReasons: isExactMatch || isCloseMatch ? ['Title match'] : ['Text match'],
      };
    }));
  }

  // Similar content search if references found (but DON'T overshadow direct matches)
  if (query.intent?.similar_to?.length) {
    // Search for the referenced content first
    for (const ref of query.intent.similar_to.slice(0, 3)) {
      const { results: refResults } = await searchMulti(ref);
      if (refResults.length > 0) {
        const firstMatch = refResults[0];

        // Add the referenced content itself with high score
        const alreadyExists = results.some(r =>
          r.content.id === firstMatch.id && r.content.mediaType === firstMatch.mediaType
        );
        if (!alreadyExists) {
          results.push({
            content: firstMatch,
            relevanceScore: 0.98, // Very high, but below exact title match
            matchReasons: ['Referenced title'],
          });
        }

        // Then get similar content
        const similar = firstMatch.mediaType === 'movie'
          ? await getSimilarMovies(firstMatch.id)
          : await getSimilarTVShows(firstMatch.id);

        results.push(...similar.map(content => ({
          content,
          relevanceScore: 0.75, // Lower than direct matches
          matchReasons: [`Similar to "${ref}"`],
        })));
      }
    }
  }

  // Discovery-based search with filters
  let genresForDiscovery = query.filters?.genres || [];
  console.log(`🎬 Initial genres for discovery: [${genresForDiscovery.join(', ')}]`);

  // If no genres mapped but keywords suggest children/family content, use family genre
  if (genresForDiscovery.length === 0) {
    const keywords = (query.intent?.keywords || []).map(k => k.toLowerCase());
    const themes = (query.intent?.themes || []).map(t => t.toLowerCase());
    const allText = [
      query.query.toLowerCase(),
      ...themes,
      ...keywords,
    ].join(' ');

    console.log(`🔍 Checking for children/family keywords in: "${allText}"`);

    if (allText.match(/\b(children|kids?|family|child|young|appropriate|age|year.?old)\b/i)) {
      console.log('🎯 Detected children/family intent, adding family genre');
      genresForDiscovery = [10751]; // Family genre ID
    }
  }

  console.log(`🎬 Final genres for discovery: [${genresForDiscovery.join(', ')}]`);

  if (genresForDiscovery.length > 0) {
    // Note: Many genre IDs work for both movies and TV (e.g., family=10751, comedy=35)
    // Don't filter by ID range - just use all genres for both movie and TV discovery
    console.log(`🎬 Running discovery with genres: [${genresForDiscovery.join(', ')}]`);

    if (query.filters?.mediaType !== 'tv') {
      const { results: discoveredMovies } = await discoverMovies({
        genres: genresForDiscovery,
        ratingMin: query.filters?.ratingMin,
      });
      console.log(`🎬 Discovered ${discoveredMovies.length} movies`);
      results.push(...discoveredMovies.map(content => ({
        content,
        relevanceScore: 0.7,
        matchReasons: ['Genre match'],
      })));
    }

    if (query.filters?.mediaType !== 'movie') {
      const { results: discoveredShows } = await discoverTVShows({
        genres: genresForDiscovery,
        ratingMin: query.filters?.ratingMin,
      });
      console.log(`🎬 Discovered ${discoveredShows.length} TV shows`);
      results.push(...discoveredShows.map(content => ({
        content,
        relevanceScore: 0.7,
        matchReasons: ['Genre match'],
      })));
    }
  }

  return results;
}

/**
 * Perform vector similarity search
 */
async function performVectorSearch(query: SemanticSearchQuery): Promise<SearchResult[]> {
  try {
    // Get embedding for the query
    const queryEmbedding = await getContentEmbedding(query.query);
    if (!queryEmbedding) return [];

    // Search vector database
    const vectorResults = await searchByEmbedding(queryEmbedding, 20);

    return vectorResults.map(result => ({
      content: result.content,
      relevanceScore: result.score,
      matchReasons: ['Semantic similarity'],
      similarityScore: result.score,
    }));
  } catch (error) {
    console.error('Vector search failed:', error);
    return [];
  }
}

/**
 * Merge and rank results from multiple sources
 */
function mergeAndRankResults(
  tmdbResults: SearchResult[],
  vectorResults: SearchResult[],
  query: SemanticSearchQuery,
  userPreferences?: number[]
): SearchResult[] {
  // Combine results, deduplicating by ID
  const resultMap = new Map<string, SearchResult>();

  // Process TMDB results
  for (const result of tmdbResults) {
    const key = `${result.content.mediaType}-${result.content.id}`;
    const existing = resultMap.get(key);
    if (existing) {
      existing.relevanceScore = Math.max(existing.relevanceScore, result.relevanceScore);
      existing.matchReasons.push(...result.matchReasons);
    } else {
      resultMap.set(key, { ...result });
    }
  }

  // Process vector results with higher weight
  for (const result of vectorResults) {
    const key = `${result.content.mediaType}-${result.content.id}`;
    const existing = resultMap.get(key);
    if (existing) {
      // Boost score significantly for vector matches
      existing.relevanceScore = Math.min(1, existing.relevanceScore + (result.similarityScore || 0) * 0.5);
      existing.matchReasons.push(...result.matchReasons);
      existing.similarityScore = result.similarityScore;
    } else {
      resultMap.set(key, { ...result, relevanceScore: (result.similarityScore || 0.5) * 0.8 });
    }
  }

  // Convert to array and apply additional scoring
  let results = Array.from(resultMap.values());

  // Apply intent-based scoring
  if (query.intent) {
    results = results.map(result => {
      let boost = 0;

      // Boost for matching themes (map themes to potential genre matches)
      if (query.intent?.themes?.length) {
        const themeGenres = query.intent.themes
          .map((t: string) => GENRE_MAP[t.toLowerCase()])
          .filter(Boolean);
        if (themeGenres.some((g: { movie?: number; tv?: number }) =>
          g?.movie && result.content.genreIds.includes(g.movie)
        )) {
          boost += 0.1;
          result.matchReasons.push('Theme match');
        }
      }

      // Popularity boost for highly rated content
      if (result.content.voteAverage >= 7.5 && result.content.voteCount > 1000) {
        boost += 0.05;
      }

      return {
        ...result,
        relevanceScore: Math.min(1, result.relevanceScore + boost),
      };
    });
  }

  // Apply user preference boost
  if (userPreferences?.length) {
    results = results.map(result => {
      const genreMatch = result.content.genreIds.some(id => userPreferences.includes(id));
      if (genreMatch) {
        return {
          ...result,
          relevanceScore: Math.min(1, result.relevanceScore + 0.1),
          matchReasons: [...result.matchReasons, 'Matches your preferences'],
        };
      }
      return result;
    });
  }

  // Sort by relevance and deduplicate match reasons
  return results
    .map(r => ({
      ...r,
      matchReasons: [...new Set(r.matchReasons)],
    }))
    .sort((a, b) => b.relevanceScore - a.relevanceScore)
    .slice(0, 50);
}

/**
 * Generate a natural language explanation for why content was recommended
 */
export async function explainRecommendation(
  content: MediaContent,
  userQuery: string,
  matchReasons: string[]
): Promise<string> {
  try {
    const { text } = await generateText({
      model: openai('gpt-4o-mini'),
      prompt: `Generate a brief, engaging explanation for why "${content.title}" was recommended to a user who searched for: "${userQuery}"

Match reasons: ${matchReasons.join(', ')}
Content overview: ${content.overview}
Rating: ${content.voteAverage}/10

Write a 1-2 sentence explanation that's conversational and highlights why this is a good match for their search. Don't mention the rating unless it's relevant.`,
    });

    return text;
  } catch (error) {
    // Fallback to simple explanation
    return matchReasons[0] || 'Based on your search';
  }
}
