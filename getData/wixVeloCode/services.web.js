import { wixEventsV2 } from 'wix-events.v2';
import { posts } from 'wix-blog-backend';
import wixData from 'wix-data';

const SERVICE_FIND_EVENTS = 'REPLACE_ME_EVENTS_SERVICE_FUNCTION';
const SERVICE_FIND_POSTS = 'REPLACE_ME_POSTS_SERVICE_FUNCTION';
const SERVICE_FIND_ARTICLES = 'REPLACE_ME_ARTICLES_SERVICE_FUNCTION';

/**
 * https://dev.wix.com/docs/velo/apis/wix-events-v2/wix-events-v2/publish-draft-event
 */
export async function yourFindEventsServiceFunction() {
	let allEvents = [];

	let query = wixEventsV2.queryEvents().ne('status', 'CANCELED').limit(50);

	try {
		let results = await query.find();

		// Loop through all pages of results
		while (results.items.length > 0) {
			allEvents = allEvents.concat(results.items); // Add the current page's items to the list

			if (results.hasNext()) {
				results = await results.next(); // Get the next page
			} else {
				break; // Exit loop if no more pages
			}
		}

		return allEvents;
	} catch (error) {
		console.error(
			`❌ [${SERVICE_FIND_EVENTS}]: Error querying events from Wix Events:`,
			error
		);
		throw new Error('Failed to retrieve all events from Wix Events V2');
	}
}

/**
 * https://dev.wix.com/docs/velo/apis/wix-blog-backend/posts/query-posts
 */
export async function yourFindPostsServiceFunction(options) {
	let allPosts = [];
	let query = posts.queryPosts(options).limit(50); // Start the query with a limit

	try {
		let results = await query.find();

		// Loop through all pages of results
		while (results.items.length > 0) {
			allPosts = allPosts.concat(results.items); // Add the current page's items to the list

			if (results.hasNext()) {
				results = await results.next(); // Get the next page
			} else {
				break; // Exit loop if no more pages
			}
		}

		return allPosts;
	} catch (error) {
		console.error(`❌ [${SERVICE_FIND_POSTS}]: Error querying post:`, error);
		throw new Error('Failed to retrieve posts.');
	}
}

/**
 * https://dev.wix.com/docs/velo/apis/wix-data/query
 */
export async function yourFindArticlesServiceFunction() {
	try {
		const retrievedCollection = await wixData.query(collectionId).find();
		return retrievedCollection;
	} catch (error) {
		console.error(
			`❌ [${SERVICE_FIND_ARTICLES}]: Error querying articles:`,
			error
		);
		throw new Error('Failed to retrieve articles.');
	}
}
