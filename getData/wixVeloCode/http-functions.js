import { ok, badRequest } from 'wix-http-functions';
import {
	validateAuthorizationHeader,
	decodeJwtToken,
	WIX_AUTH_SECRET,
} from 'backend/util.web';
import {
	yourFindEventsServiceFunction,
	yourFindPostsServiceFunction,
	yourFindArticlesServiceFunction,
} from 'backend/services.web';

const ENDPOINT_FIND_EVENTS = 'get_REPLACE_ME_EVENTS_ENDPOINT';
const ENDPOINT_FIND_POSTS = 'get_REPLACE_ME_POSTS_ENDPOINT';
const ENDPOINT_FIND_ARTICLES = 'get_REPLACE_ME_ARTICLES_ENDPOINT';

const EXPECTED_JWT_SUB_EVENTS = 'REPLACE-ME-EVENTS-JWT-SUB';
const EXPECTED_JWT_SUB_POSTS = 'REPLACE-ME-POSTS-JWT-SUB';
const EXPECTED_JWT_SUB_ARTICLES = 'REPLACE-ME-ARTICLES-JWT-SUB';

/**
 * HTTP Function: get_yourEventsEndpoint
 * Endpoint: https://www.yourdomain.com/_functions/yourEventsEndpoint
 *
 * @accepts GET with JSON body:
 * @requires Authorization header: `Bearer YOUR_JWT_TOKEN`
 * @returns {object} Events data (array of events objects) if successful, or an error object.
 */
export async function get_yourEventsEndpoint(request) {
	const response = { headers: { 'Content-Type': 'application/json' } };

	try {
		const actualToken = validateAuthorizationHeader(
			request,
			ENDPOINT_FIND_EVENTS
		);

		const decodedJwt = await decodeJwtToken(
			actualToken,
			WIX_AUTH_SECRET,
			EXPECTED_JWT_SUB_EVENTS,
			ENDPOINT_FIND_EVENTS
		);

		console.log('DEBUG: Actual Token: ', actualToken);
		console.log('DEBUG: Decoded JWT: ', decodedJwt);

		if (!decodedJwt) {
			console.error(`❌ [${ENDPOINT_FIND_EVENTS}]: Invalid or missing JWT.`);
			response.body = { error: 'Unauthorized: Invalid token.' };
			return badRequest(response);
		}

		const eventsResult = await yourFindEventsServiceFunction();

		console.log(
			`🌐 [${ENDPOINT_FIND_EVENTS}]: Received events: `,
			eventsResult.length ?? 0
		);

		response.body = eventsResult;
		return ok(response);
	} catch (error) {
		console.error(
			`❌ [${ENDPOINT_FIND_EVENTS}]: Unexpected error caught: `,
			error
		);
		response.body = { error: error?.message || String(error) };
		return badRequest(response);
	}
}

/**
 * HTTP Function: get_yourPostsEndpoint
 * Endpoint: https://www.yourdomain.com/_functions/yourPostsEndpoint
 *
 * @accepts GET with JSON body:
 * @requires Authorization header: `Bearer YOUR_JWT_TOKEN`
 * @returns {object} Posts data (array of posts objects) if successful, or an error object.
 */
export async function get_yourPostsEndpoint(request) {
	const response = { headers: { 'Content-Type': 'application/json' } };

	const options = {
		fieldsets: ['CONTENT_TEXT', 'URL', 'METRICS'],
	};

	try {
		const actualToken = validateAuthorizationHeader(
			request,
			ENDPOINT_FIND_POSTS
		);

		const decodedJwt = await decodeJwtToken(
			actualToken,
			WIX_AUTH_SECRET,
			EXPECTED_JWT_SUB_POSTS,
			ENDPOINT_FIND_POSTS
		);

		console.log('DEBUG: Actual Token: ', actualToken);
		console.log('DEBUG: Decoded JWT: ', decodedJwt);

		if (!decodedJwt) {
			console.error(`❌ [${ENDPOINT_FIND_EVENTS}]: Invalid or missing JWT.`);
			response.body = { error: 'Unauthorized: Invalid token.' };
			return badRequest(response);
		}

		const postsResult = await yourFindPostsServiceFunction(options);

		console.log(
			`🌐 [${ENDPOINT_FIND_POSTS}]: Received posts: `,
			postsResult.length ?? 0
		);

		response.body = postsResult;
		return ok(response);
	} catch (error) {
		console.error(
			`❌ [${ENDPOINT_FIND_POSTS}]: Unexpected error caught: `,
			error
		);
		response.body = { error: error?.message || String(error) };
		return badRequest(response);
	}
}

/**
 * HTTP Function: get_yourArticlesEndpoint
 * Endpoint: https://www.yourdomain.com/_functions/yourArticlesEndpoint
 *
 * @accepts GET with JSON body:
 * @requires Authorization header: `Bearer YOUR_JWT_TOKEN`
 * @returns {object} Articles data (array of articles objects) if successful, or an error object.
 */
export async function get_yourArticlesEndpoint(request) {
	const response = { headers: { 'Content-Type': 'application/json' } };

	const collectionId = 'REPLACE-ME-YOUR-COLLECTION-ID';

	try {
		const actualToken = validateAuthorizationHeader(
			request,
			ENDPOINT_FIND_ARTICLES
		);

		const decodedJwt = await decodeJwtToken(
			actualToken,
			WIX_AUTH_SECRET,
			EXPECTED_JWT_SUB_ARTICLES,
			ENDPOINT_FIND_ARTICLES
		);

		console.log('DEBUG: Actual Token: ', actualToken);
		console.log('DEBUG: Decoded JWT: ', decodedJwt);

		if (!decodedJwt) {
			console.error(`❌ [${ENDPOINT_FIND_EVENTS}]: Invalid or missing JWT.`);
			response.body = { error: 'Unauthorized: Invalid token.' };
			return badRequest(response);
		}

		const articlesResult = await yourFindArticlesServiceFunction(collectionId);

		console.log(
			`🌐 [${ENDPOINT_FIND_ARTICLES}]: Received articles: `,
			articlesResult.length ?? 0
		);

		response.body = articlesResult;
		return ok(response);
	} catch (error) {
		console.error(
			`❌ [${ENDPOINT_FIND_ARTICLES}]: Unexpected error caught: `,
			error
		);
		response.body = { error: error?.message || String(error) };
		return badRequest(response);
	}
}
