<script lang="ts">
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';

	// Widgets
	import Dock from '$lib/components/dock.svelte';

	let { children } = $props();
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>

{@render children()}
<Dock />
